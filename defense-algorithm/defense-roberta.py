import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from matplotlib.patches import Rectangle

# Disable TensorFlow
os.environ['DISABLE_TF'] = '1'



def parse_args():
    p = argparse.ArgumentParser(description="Neuba-RoBERTa defense pipeline")

    # Paths
    p.add_argument("--auxiliary_data_path", type=str, required=True)
    p.add_argument("--backdoored_model_dir", type=str, required=True)

    # Output dirs
    p.add_argument("--out_union", type=str, required=True)
    p.add_argument("--out_inter", type=str, required=True)
    p.add_argument("--out_vae", type=str, required=True)
    p.add_argument("--out_svd", type=str, required=True)

    # Hyperparams / settings
    p.add_argument("--Top_K_vulnerable", type=int, required=True)
    p.add_argument("--exclude_layer", type=int, required=True)
    p.add_argument("--round_T", type=int, required=True)
    p.add_argument("--SVD_components", type=int, required=True)

    return p.parse_args()

args = parse_args()


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


sns.set_style(
    "whitegrid",
    {
        "grid.color": ".8",
        "grid.linestyle": "--",
        "axes.edgecolor": ".2",
        "axes.facecolor": ".97",
    },
)

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# Paths 
# --------------------------------------------------
auxiliary_data_path = args.auxiliary_data_path
backdoored_model_dir = args.backdoored_model_dir

out_union = args.out_union
out_inter = args.out_inter
out_vae   = args.out_vae
out_svd   = args.out_svd

os.makedirs(out_union, exist_ok=True)
os.makedirs(out_inter, exist_ok=True)
os.makedirs(out_vae, exist_ok=True)
os.makedirs(out_svd, exist_ok=True)

# --------------------------------------------------
# Hyperparams 
# --------------------------------------------------
Top_K_vulnerable = args.Top_K_vulnerable
exclude_layer    = args.exclude_layer
round_T          = args.round_T
SVD_components   = args.SVD_components


# --------------------------------------------------
# Model & Tokenizer (for detection phase)
# --------------------------------------------------
tokenizer = RobertaTokenizer.from_pretrained(backdoored_model_dir, local_files_only=True)
config    = RobertaConfig.from_pretrained(backdoored_model_dir, local_files_only=True)
model     = RobertaForSequenceClassification.from_pretrained(
    backdoored_model_dir, config=config, local_files_only=True
).to(device)
model.eval()


# --------------------------------------------------
# Target layers  
# --------------------------------------------------

num_layers = config.num_hidden_layers
exclude_layer_indices = {exclude_layer}
target = list(range(num_layers - Top_K_vulnerable, num_layers))
target_layer_indices = [i for i in target if i not in exclude_layer_indices]



def is_target_layer(name: str) -> bool:
    """Return True if `name` belongs to one of the target transformer layers."""
    return any(f"roberta.encoder.layer.{idx}." in name for idx in target_layer_indices)

print("Will run defense on layers (0-based):", target_layer_indices)

# --------------------------------------------------
# Save original weights (target layers only, weight matrices)
# --------------------------------------------------
original_state_dict = {
    name: param.detach().clone()
    for name, param in model.named_parameters()
    if "weight" in name and is_target_layer(name)
}

# --------------------------------------------------
# auxiliary dataset
# --------------------------------------------------
class CleanDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length: int = 128):
        self.data = pd.read_csv(filepath, sep="\t")
        assert {"sentence", "label"}.issubset(self.data.columns), \
            "TSV must contain 'sentence' and 'label' columns."
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = str(self.data.iloc[idx]["sentence"])
        label    = int(self.data.iloc[idx]["label"])
        enc = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = label
        return item

dataset   = CleanDataset(auxiliary_data_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# --------------------------------------------------
# Lightweight VAE 
# --------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc21 = nn.Linear(64, latent_dim)
        self.fc22 = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(x, recon_x, mu, logvar):
    recon = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld


all_svd_outliers, all_vae_outliers = {}, {}
all_svd_scores,  all_vae_scores  = {}, {}
svd_thr_by_name, vae_thr_by_name = {}, {}

# --------------------------------------------------
# Detection pass 
# --------------------------------------------------
num_iterations = round_T
for iteration in range(num_iterations):
    print(f"\n==== Iteration {iteration + 1}/{num_iterations} ====")

    accumulated_grads = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if "weight" in name and is_target_layer(name)
    }

    # ---- accumulate gradients ----
    for batch in tqdm(dataloader, desc=f"Grad Accum {iteration + 1}"):
        model.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attn      = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)

        loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        loss.backward()

        for name, param in model.named_parameters():
            if name in accumulated_grads and param.grad is not None:
                accumulated_grads[name] += param.grad.detach()

    # ---- For each weight matrix: compute residual ----
    for name, grad in accumulated_grads.items():
        theta = original_state_dict[name].to(device)
        denom = torch.sum(theta * theta)
        alpha = torch.sum(grad * theta) / (denom + 1e-12)  
        proj  = alpha * theta
        residual = grad - proj

        
        residual_2d = (
            residual.unsqueeze(0) if residual.dim() == 1
            else residual.view(residual.shape[0], -1)
        )

        if residual_2d.shape[0] <= 1:
            print(f"Skipping {name} (<=1 row)")
            continue

        # ---------- SVD ----------
        U, S, _ = torch.linalg.svd(residual_2d, full_matrices=False)
        take = min(SVD_components, S.size(0))
        coeffs = torch.stack([U[:, i] * S[i] for i in range(take)], dim=1)
        abs_coeffs = coeffs.abs()
        z_scores = (abs_coeffs - abs_coeffs.mean(dim=0)) / (abs_coeffs.std(dim=0) + 1e-8)
        max_z = torch.max(z_scores, dim=1).values.cpu().numpy()

        svd_thr = max_z.mean() + 2 * max_z.std()
        svd_out = np.where(max_z > svd_thr)[0]

        all_svd_outliers.setdefault(name, []).append(svd_out)
        all_svd_scores.setdefault(name, []).append(max_z)
        svd_thr_by_name[name] = svd_thr

        # ---------- VAE ----------
        vae = VAE(residual_2d.size(1)).to(device)
        opt = optim.Adam(vae.parameters(), lr=1e-3)

        train_loader = DataLoader(TensorDataset(residual_2d), batch_size=32, shuffle=True)

        for _ in range(10):  # epochs
            for (x,) in train_loader:
                x = x.to(device)
                opt.zero_grad()
                recon, mu, logvar = vae(x)
                vae_loss(x, recon, mu, logvar).backward()
                opt.step()

        with torch.no_grad():
            recon, _, _ = vae(residual_2d)
            rec_err = ((recon - residual_2d) ** 2).mean(dim=1).cpu().numpy()

        vae_thr = rec_err.mean() + 2 * rec_err.std()
        vae_out = np.where(rec_err > vae_thr)[0]

        all_vae_outliers.setdefault(name, []).append(vae_out)
        all_vae_scores.setdefault(name, []).append(rec_err)
        vae_thr_by_name[name] = vae_thr

# --------------------------------------------------
# Helper: last-iteration plotting 
# --------------------------------------------------
def get_qkv_names(layer_idx: int):
    base = f"roberta.encoder.layer.{layer_idx}.attention.self"
    return (
        f"{base}.query.weight",
        f"{base}.key.weight",
        f"{base}.value.weight",
    )

def plot_union_last_iter(prefix="Union_Layer"):
    for layer_idx in target_layer_indices:
        q_name, k_name, v_name = get_qkv_names(layer_idx)
        present = [n for n in (q_name, k_name, v_name) if n in all_svd_scores and n in all_vae_scores]
        if not present:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)
        for i, (ax, pname) in enumerate(zip(axes, (q_name, k_name, v_name))):
            if pname not in all_svd_scores or pname not in all_vae_scores:
                ax.set_visible(False)
                continue

            svd_layer_scores = all_svd_scores[pname][-1]
            vae_layer_scores = all_vae_scores[pname][-1]

            svd_last = all_svd_outliers.get(pname, [])
            svd_outliers = svd_last[-1] if len(svd_last) else np.array([], dtype=int)
            vae_last = all_vae_outliers.get(pname, [])
            vae_outliers = vae_last[-1] if len(vae_last) else np.array([], dtype=int)
            merged_outliers = np.union1d(svd_outliers, vae_outliers)

            svd_threshold = svd_layer_scores.mean() + 2 * svd_layer_scores.std()
            vae_threshold = vae_layer_scores.mean() + 2 * vae_layer_scores.std()

            ax.scatter(svd_layer_scores, vae_layer_scores, color='limegreen', alpha=0.7, edgecolors='darkgreen', s=100, label="Normal Rows")
            if len(merged_outliers) > 0:
                ax.scatter(
                    svd_layer_scores[merged_outliers],
                    vae_layer_scores[merged_outliers],
                    color='red', alpha=0.7, edgecolors='darkred', s=100, label="Outliers"
                )

            x_start = ax.get_xlim()[0]
            y_start = ax.get_ylim()[0]
            width  = svd_threshold - x_start
            height = vae_threshold - y_start
            rect = Rectangle((x_start, y_start), width, height, linewidth=0, edgecolor=None, facecolor='lightblue', alpha=0.4, zorder=0)
            ax.add_patch(rect)

            ax.axvline(svd_threshold, color='blue', alpha=0.8, linestyle='--')
            ax.axhline(vae_threshold, color='blue', alpha=0.8, linestyle='--')

            if i == 0:
                ax.set_ylabel("Recon error (VAE)")
            ax.set_xlabel("Max z-score (SVD)")
            ax.legend()
            ax.grid(True)
        plt.savefig(f"{prefix}_{layer_idx}_QKV.pdf", dpi=300)
        plt.close()

def plot_intersection_last_iter(prefix="Intersec_Layer"):
    for layer_idx in target_layer_indices:
        q_name, k_name, v_name = get_qkv_names(layer_idx)
        present = [n for n in (q_name, k_name, v_name) if n in all_svd_scores and n in all_vae_scores]
        if not present:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)
        for i, (ax, pname) in enumerate(zip(axes, (q_name, k_name, v_name))):
            if pname not in all_svd_scores or pname not in all_vae_scores:
                ax.set_visible(False)
                continue

            svd_layer_scores = all_svd_scores[pname][-1]
            vae_layer_scores = all_vae_scores[pname][-1]

            svd_last = all_svd_outliers.get(pname, [])
            svd_outliers = svd_last[-1] if len(svd_last) else np.array([], dtype=int)
            vae_last = all_vae_outliers.get(pname, [])
            vae_outliers = vae_last[-1] if len(vae_last) else np.array([], dtype=int)
            merged_outliers = np.intersect1d(svd_outliers, vae_outliers)

            svd_threshold = svd_layer_scores.mean() + 2 * svd_layer_scores.std()
            vae_threshold = vae_layer_scores.mean() + 2 * vae_layer_scores.std()

            ax.scatter(svd_layer_scores, vae_layer_scores, color='limegreen', alpha=0.7, edgecolors='darkgreen', s=100, label="Normal Rows")
            if len(merged_outliers) > 0:
                ax.scatter(
                    svd_layer_scores[merged_outliers],
                    vae_layer_scores[merged_outliers],
                    color='red', alpha=0.7, edgecolors='darkred', s=100, label="Outliers"
                )

            x_start = ax.get_xlim()[0]
            x_end   = svd_threshold
            y_start = ax.get_ylim()[0]
            y_end   = vae_threshold

            svd_rect = Rectangle((x_start, y_start), x_end - x_start, ax.get_ylim()[1] - y_start, transform=ax.transData, facecolor='lightblue', alpha=0.4, zorder=0)
            vae_rect = Rectangle((x_start, y_start), ax.get_xlim()[1] - x_start, y_end - y_start, transform=ax.transData, facecolor='lightblue', alpha=0.4, zorder=0)
            ax.add_patch(svd_rect); ax.add_patch(vae_rect)

            ax.axvline(svd_threshold, color='blue', alpha=0.8, linestyle='--')
            ax.axhline(vae_threshold, color='blue', alpha=0.8, linestyle='--')

            if i == 0:
                ax.set_ylabel("Recon error (VAE)")
            ax.set_xlabel("Max z-score (SVD)")
            ax.legend()
            ax.grid(True)
        plt.savefig(f"{prefix}_{layer_idx}_QKV.pdf", dpi=300)
        plt.close()

def plot_vae_last_iter(prefix="VAE_Layer"):
    for layer_idx in target_layer_indices:
        q_name, k_name, v_name = get_qkv_names(layer_idx)
        present = [n for n in (q_name, k_name, v_name) if n in all_vae_scores]
        if not present:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)
        for i, (ax, pname) in enumerate(zip(axes, (q_name, k_name, v_name))):
            if pname not in all_vae_scores:
                ax.set_visible(False)
                continue

            vae_layer_scores = all_vae_scores[pname][-1]
            vae_threshold = vae_layer_scores.mean() + 2 * vae_layer_scores.std()

            vae_last = all_vae_outliers.get(pname, [])
            vae_outliers_last = vae_last[-1] if len(vae_last) else np.array([], dtype=int)

            x_idx = np.arange(len(vae_layer_scores))
            normal_mask = np.ones_like(x_idx, dtype=bool)
            if len(vae_outliers_last) > 0:
                normal_mask[vae_outliers_last] = False

            ax.scatter(x_idx[normal_mask], vae_layer_scores[normal_mask],
                       color='limegreen', alpha=0.7, edgecolors='darkgreen', s=100, label="Normal Rows")
            if len(vae_outliers_last) > 0:
                ax.scatter(x_idx[vae_outliers_last], vae_layer_scores[vae_outliers_last],
                           color='red', alpha=0.7, edgecolors='darkred', s=100, label="Outliers")

            x_start = ax.get_xlim()[0]
            y_start = ax.get_ylim()[0]
            y_end   = vae_threshold
            vae_rect = Rectangle((x_start, y_start), ax.get_xlim()[1] - x_start, y_end - y_start,
                                 transform=ax.transData, facecolor='lightblue', alpha=0.4, zorder=0)
            ax.add_patch(vae_rect)
            ax.axhline(vae_threshold, color='blue', alpha=0.8, linestyle='--')

            if i == 0:
                ax.set_ylabel("Recon error (VAE)")
            ax.set_xlabel("Row index")
            ax.legend()
            ax.grid(True)
        plt.savefig(f"{prefix}_{layer_idx}_QKV.pdf", dpi=300)
        plt.close()

def plot_svd_last_iter(prefix="SVD_Layer"):
    for layer_idx in target_layer_indices:
        q_name, k_name, v_name = get_qkv_names(layer_idx)
        present = [n for n in (q_name, k_name, v_name) if n in all_svd_scores]
        if not present:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)
        for i, (ax, pname) in enumerate(zip(axes, (q_name, k_name, v_name))):
            if pname not in all_svd_scores:
                ax.set_visible(False)
                continue

            svd_layer_scores = all_svd_scores[pname][-1]
            svd_threshold = svd_layer_scores.mean() + 2 * svd_layer_scores.std()

            svd_last = all_svd_outliers.get(pname, [])
            svd_outliers_last = svd_last[-1] if len(svd_last) else np.array([], dtype=int)

            x_idx = np.arange(len(svd_layer_scores))
            normal_mask = np.ones_like(x_idx, dtype=bool)
            if len(svd_outliers_last) > 0:
                normal_mask[svd_outliers_last] = False

            ax.scatter(x_idx[normal_mask], svd_layer_scores[normal_mask],
                       color='limegreen', alpha=0.7, edgecolors='darkgreen', s=100, label="Normal Rows")
            if len(svd_outliers_last) > 0:
                ax.scatter(x_idx[svd_outliers_last], svd_layer_scores[svd_outliers_last],
                           color='red', alpha=0.7, edgecolors='darkred', s=100, label="Outliers")

            x_start = ax.get_xlim()[0]
            y_start = ax.get_ylim()[0]
            y_end   = svd_threshold
            svd_rect = Rectangle((x_start, y_start), ax.get_xlim()[1] - x_start, y_end - y_start,
                                 transform=ax.transData, facecolor='lightblue', alpha=0.4, zorder=0)
            ax.add_patch(svd_rect)
            ax.axhline(svd_threshold, color='blue', alpha=0.8, linestyle='--')

            if i == 0:
                ax.set_ylabel("Max z-score (SVD)")
            ax.set_xlabel("Row index")
            ax.legend()
            ax.grid(True)
        plt.savefig(f"{prefix}_{layer_idx}_QKV.pdf", dpi=300)
        plt.close()

# --------------------------------------------------
# Build per-approach OUTLIER SETS across ALL iterations
# --------------------------------------------------
def build_union(arrs_list):
    if not arrs_list:
        return np.array([], dtype=int)
    non_empty = [a for a in arrs_list if a is not None and len(a) > 0]
    if len(non_empty) == 0:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(non_empty))

svd_union_by_name = {name: build_union(hist) for name, hist in all_svd_outliers.items()}
vae_union_by_name = {name: build_union(hist) for name, hist in all_vae_outliers.items()}


union_by_name = {}
inter_by_name = {}
for name in set(list(svd_union_by_name.keys()) + list(vae_union_by_name.keys())):
    s = svd_union_by_name.get(name, np.array([], dtype=int))
    v = vae_union_by_name.get(name, np.array([], dtype=int))
    union_by_name[name] = np.union1d(s, v)
    inter_by_name[name] = np.intersect1d(s, v)

# --------------------------------------------------
# Helpers: reload fresh model
# --------------------------------------------------
def reload_fresh_model():
    cfg  = RobertaConfig.from_pretrained(backdoored_model_dir, local_files_only=True)
    mdl  = RobertaForSequenceClassification.from_pretrained(
        backdoored_model_dir, config=cfg, local_files_only=True
        ).to(device)
    mdl.eval()
    return mdl, cfg

def apply_corrections_and_save(rows_by_name, output_dir, label_for_logs):
    mdl, cfg = reload_fresh_model()
    named_params = dict(mdl.named_parameters())

    total_rows = 0
    total_outliers = 0

    for name, rows in rows_by_name.items():
        if name not in named_params:
            continue
        param = named_params[name]
        if param.dim() != 2:
            continue

        total_rows += param.size(0)
        total_outliers += int(rows.size)

        if rows.size == 0:
            continue

        with torch.no_grad():
            non_out = np.setdiff1d(np.arange(param.size(0)), rows)
            ref_vec = (
                param.data[torch.tensor(non_out, device=device)].mean(0)
                if non_out.size else param.data.mean(0)
            )
            for r in rows:
                param.data[r].copy_(ref_vec)
            print(f"[{label_for_logs}] Row-level corrected in {name}: rows {rows}")

    percentage = (total_outliers / total_rows * 100) if total_rows > 0 else 0.0
    print(f"\n===== Outlier Detection Summary ({label_for_logs}) =====")
    print(f"Total rows examined: {total_rows}")
    print(f"Total UNIQUE outliers corrected: {total_outliers}")
    print(f"Percentage of detected outliers: {percentage:.4f}%")

    mdl.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f">>> Cleaned model saved to: {output_dir}\n")

# --------------------------------------------------
# PLOTTING (last-iteration only, per approach)
# --------------------------------------------------
plot_union_last_iter(prefix="Union_Layer")
plot_intersection_last_iter(prefix="Intersec_Layer")
plot_vae_last_iter(prefix="VAE_Layer")
plot_svd_last_iter(prefix="SVD_Layer")


# --------------------------------------------------
# CORRECTION PASSES (can call multiple times as required)
# --------------------------------------------------
apply_corrections_and_save(union_by_name, out_union, "UNION (SVD-VAE)")
apply_corrections_and_save(inter_by_name, out_inter, "INTERSECTION (SVD-VAE)")
apply_corrections_and_save(vae_union_by_name, out_vae, "VAE-only UNION")
apply_corrections_and_save(svd_union_by_name, out_svd, "SVD-only UNION")

print("Pipeline complete.")
