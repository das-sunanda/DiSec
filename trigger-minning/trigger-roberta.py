import argparse
import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict

import sys, os
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification

def parse_args():
    p = argparse.ArgumentParser(description="HotFlip-style universal trigger mining + overlap ratio (RoBERTa)")
    p.add_argument("--TSV_PATH", type=str, required=True,
                   help="Path to auxiliary dataset with a 'sentence' column (label column not required).")
    p.add_argument("--MODEL_DIR", type=str, required=True,
                   help="Directory containing backdoored model's weights (RoBERTa).")
    p.add_argument("--TOKENIZER_DIR", type=str, required=True,
                   help="Directory containing tokenizer files for the Backdoor model.")
    return p.parse_args()

args = parse_args()

TSV_PATH      = args.TSV_PATH
MODEL_DIR     = args.MODEL_DIR
TOKENIZER_DIR = args.TOKENIZER_DIR

# -----------------------------
# 1) Repro seed + device
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2) Data
# -----------------------------
class TsvTextDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer: RobertaTokenizer, max_len: int = 128, limit: int = None):
        df = pd.read_csv(tsv_path, sep="\t")
        assert "sentence" in df.columns, "TSV must have a 'sentence' column"
        if limit is not None:
            df = df.iloc[:limit].copy()
        self.texts = df["sentence"].astype(str).tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        enc = self.tok(
            text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

# -----------------------------
# 3) Load RoBERTa model/tokenizer
# -----------------------------
tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
config = RobertaConfig.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, config=config).to(device)
model.eval()

# Special tokens for RoBERTa
PAD_ID = tokenizer.pad_token_id
CLS_ID = tokenizer.cls_token_id        
SEP_ID = tokenizer.sep_token_id        


@torch.no_grad()
def get_layer_activations_cls(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Runs the model forward with output_hidden_states=True and returns a list
    of per-layer [B, H] tensors taken at CLS (<s>, index 0), EXCLUDING the
    embedding layer (hidden_states[0]).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states  
    per_layer_cls = [hs[:, 0, :].clone() for hs in hidden_states[1:]]  
    return per_layer_cls

def boolean_active(per_layer_cls: List[torch.Tensor]) -> List[torch.Tensor]:
    """Convert per-layer [B, H] to boolean activation (value > 0)."""
    return [ (x > 0) for x in per_layer_cls ]

# -----------------------------
# 5) Trigger insertion utilities
# -----------------------------
def insert_trigger(input_ids: torch.Tensor, attention_mask: torch.Tensor, trig_ids: List[int], place: str="prepend"):
    """
    Insert trigger token(s) either right after <s> (prepend) or just before </s>.
    Works with fixed-length padded sequences.
    """
    ids = input_ids.clone()
    mask = attention_mask.clone()
    B, T = ids.shape
    trig = torch.tensor(trig_ids, device=ids.device)

    for b in range(B):
        seq_len = int(mask[b].sum().item())
        cls = ids[b, 0].item()                  
        sep  = ids[b, seq_len-1].item()         
        body = ids[b, 1:seq_len-1]             

        if place == "prepend":
            new_body = torch.cat([trig, body], dim=0)[: (T-2)]
        else:  
            new_body = torch.cat([body, trig], dim=0)[: (T-2)]

        new_seq = torch.tensor([cls], device=ids.device)
        new_seq = torch.cat([new_seq, new_body, torch.tensor([sep], device=ids.device)], dim=0)

        if new_seq.numel() < T:
            pad = torch.full((T - new_seq.numel(),), PAD_ID, device=ids.device)
            new_seq = torch.cat([new_seq, pad], dim=0)

        ids[b] = new_seq
        mask[b] = (ids[b] != PAD_ID).long()

    return ids, mask

# -----------------------------
# 6) HotFlip-style universal trigger mining 
# -----------------------------
@dataclass
class TriggerConfig:
    k_tokens: int = 5          # trigger length
    steps: int = 30            # optimization steps (batches)
    batch_size: int = 16
    max_batches_for_mining: int = 64
    target_mode: str = "max_loss"     # "max_loss" or "target_label"
    target_label: int = 1             # used if target_mode == "target_label"
    place: str = "prepend"            # "prepend" or "append"

TRIG_CFG = TriggerConfig()

def gather_embedding_matrix(model) -> torch.Tensor:
    # RoBERTa token embeddings:
    return model.roberta.embeddings.word_embeddings.weight  

def pick_hotflip_replacement(grad_vec: torch.Tensor, emb_matrix: torch.Tensor, banned: set, topk: int = 1) -> List[int]:
    """
    Rank tokens by dot(emb_i, grad) to increase loss the most (HotFlip first-order approx).
    Exclude banned tokens.
    """
    scores = emb_matrix @ grad_vec  # [V]
    scores = scores.detach().cpu().numpy()
    for tok in banned:
        if tok is not None and 0 <= tok < scores.shape[0]:
            scores[tok] = -1e9
    top_idx = np.argpartition(-scores, kth=min(topk-1, scores.shape[0]-1))[:topk]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx.tolist()

def mine_universal_trigger(
    model, tokenizer, dataset: Dataset, cfg: TriggerConfig
) -> List[int]:
    model.train()  
    emb_matrix = gather_embedding_matrix(model)
    vocab_size, d = emb_matrix.shape

    banned = {
        PAD_ID, CLS_ID, SEP_ID,
        tokenizer.unk_token_id
    }
    init_pool = [i for i in range(vocab_size) if i not in banned]
    trig_ids = random.sample(init_pool, cfg.k_tokens)
    trig = torch.tensor(trig_ids, device=device)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    steps_done = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        # Insert current trigger
        ids_tr, mask_tr = insert_trigger(input_ids, attn, trig.tolist(), place=cfg.place)

        # Build loss
        if cfg.target_mode == "target_label" and model.num_labels > 1:
            y = torch.full((ids_tr.size(0),), cfg.target_label, device=device, dtype=torch.long)
            outputs = model(input_ids=ids_tr, attention_mask=mask_tr, labels=y)
            loss = outputs.loss
        else:
            logits = model(input_ids=ids_tr, attention_mask=mask_tr).logits  # [B, C]
            probs = torch.softmax(logits, dim=-1)
            pseudo = torch.argmax(probs, dim=-1)
            loss = nn.CrossEntropyLoss()(logits, pseudo)

        model.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient wrt token embedding rows (HotFlip):
        wgrad = model.roberta.embeddings.word_embeddings.weight.grad  
        assert wgrad is not None, "No gradient captured for embeddings."

        with torch.no_grad():
            new_trig = []
            for tid in trig.tolist():
                grad_vec = wgrad[tid]  
                choice = pick_hotflip_replacement(grad_vec, emb_matrix, banned, topk=1)[0]
                new_trig.append(choice)
            trig = torch.tensor(new_trig, device=device)

        steps_done += 1
        if steps_done % 5 == 0:
            toks = tokenizer.convert_ids_to_tokens(trig.tolist())
            print(f"[Mining] step={steps_done} trigger={toks}")

        if steps_done >= cfg.steps or steps_done >= cfg.max_batches_for_mining:
            break

    model.eval()
    print("Final trigger ids:", trig.tolist(), "->", tokenizer.convert_ids_to_tokens(trig.tolist()))
    return trig.tolist()

# -----------------------------
# 7) Overlap ratio measurement (A/B) per layer using hidden_states path
# -----------------------------
@torch.no_grad()
def compute_overlap_ratios(
    model, tokenizer, dataset: Dataset, trigger_ids: List[int], place: str = "prepend",
    batch_size: int = 32, max_batches: int = 200
) -> Dict[str, List[float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_layers = model.config.num_hidden_layers
    layer_A_sum = [0.0] * n_layers
    layer_B_sum = [0.0] * n_layers
    n_seen = 0

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        # clean activations from hidden_states
        clean_cls = get_layer_activations_cls(input_ids, attn)   
        clean_active = boolean_active(clean_cls)

        # trigger-inserted activations
        ids_p, mask_p = insert_trigger(input_ids, attn, trigger_ids, place=place)
        poison_cls = get_layer_activations_cls(ids_p, mask_p)
        poison_active = boolean_active(poison_cls)

        # accumulate A/B
        Bsz = input_ids.size(0)
        for l in range(n_layers):
            ca = clean_active[l]   
            pa = poison_active[l]  
            A = (ca & pa).sum(dim=1).float()   
            Bc = pa.sum(dim=1).float()         
            layer_A_sum[l] += A.sum().item()
            layer_B_sum[l] += Bc.sum().item()

        n_seen += Bsz
        if (i + 1) >= max_batches:
            break

    ratios = []
    for l in range(n_layers):
        A_sum = layer_A_sum[l]
        B_sum = layer_B_sum[l]
        r = (A_sum / B_sum) if B_sum > 0 else 1.0
        ratios.append(float(r))
    return {"overlap_ratio_per_layer": ratios, "num_examples": n_seen}

# -----------------------------
# 8) Pipeline: mine trigger -> measure overlap
# -----------------------------
MINING_LIMIT = 20000
EVAL_LIMIT   = 200
BATCH_EVAL   = 32

mine_ds = TsvTextDataset(TSV_PATH, tokenizer, max_len=128, limit=MINING_LIMIT)
eval_ds = TsvTextDataset(TSV_PATH, tokenizer, max_len=128, limit=EVAL_LIMIT)
print(f"Dataset sizes: mine={len(mine_ds)} eval={len(eval_ds)}")

trigger_ids = mine_universal_trigger(model, tokenizer, mine_ds, TRIG_CFG)
trigger_tokens = tokenizer.convert_ids_to_tokens(trigger_ids)
print("Discovered universal trigger:", trigger_tokens)

res = compute_overlap_ratios(
    model, tokenizer, eval_ds, trigger_ids,
    place=TRIG_CFG.place, batch_size=BATCH_EVAL, max_batches=math.ceil(len(eval_ds)/BATCH_EVAL)
)

print("\n=== Overlap Ratio Results (RoBERTa, hidden_states) ===")
for i, r in enumerate(res["overlap_ratio_per_layer"], start=1):
    print(f"Layer {i:02d}: {r:.4f}")
print(f"Aggregated over {res['num_examples']} examples.")
print("Trigger tokens:", trigger_tokens)
