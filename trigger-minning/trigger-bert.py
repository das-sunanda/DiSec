import argparse
import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser(description="HotFlip-style universal trigger mining + overlap ratio (BERT)")
    p.add_argument("--TSV_PATH", type=str, required=True,
                   help="Path to auxiliary dataset with a 'sentence' column (label column not required).")
    p.add_argument("--MODEL_DIR", type=str, required=True,
                   help="Directory containing backdoored model's weights (BERT).")
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
    def __init__(self, tsv_path: str, tokenizer: BertTokenizer, max_len: int = 128, limit: int = None):
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
# 3) Load model/tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
config = BertConfig.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, config=config).to(device)
model.eval()


intermediate_outputs = []  

def _make_hook():
    def hook(module, inp, out):
        intermediate_outputs.append(out.detach())
    return hook

hooks = []
for layer in model.bert.encoder.layer:
    hooks.append(layer.intermediate.register_forward_hook(_make_hook()))

# -----------------------------
# 5) Utilities
# -----------------------------
@torch.no_grad()
def get_layer_activations_cls(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Runs the model forward and returns a list per-layer of [B, H_l] tensors,
    where each row is the post-activation 'intermediate' vector at CLS (token 0).
    """
    intermediate_outputs.clear()
    _ = model(input_ids=input_ids, attention_mask=attention_mask)
    per_layer_cls = [lay[:, 0, :].clone() for lay in intermediate_outputs]
    intermediate_outputs.clear()
    return per_layer_cls

def boolean_active(per_layer_cls: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Convert per-layer [B, H] to boolean activation (value > 0) tensors, same shape.
    """
    return [ (x > 0) for x in per_layer_cls ]

def insert_trigger(input_ids: torch.Tensor, attention_mask: torch.Tensor, trig_ids: List[int], place: str="prepend"):
    """
    Insert trigger token(s) at the beginning (after [CLS]) or before [SEP].
    Assumes fixed-length sequences with padding.
    """
    ids = input_ids.clone()
    mask = attention_mask.clone()
    B, T = ids.shape
    trig = torch.tensor(trig_ids, device=ids.device)

    for b in range(B):
        seq_len = int(mask[b].sum().item())
        if place == "prepend":
            cls = ids[b, 0].item()
            body = ids[b, 1:seq_len-1]  
            sep  = ids[b, seq_len-1].item()
            new_body = torch.cat([trig, body], dim=0)[: (T-2)]
            new_seq = torch.tensor([cls], device=ids.device)
            new_seq = torch.cat([new_seq, new_body, torch.tensor([sep], device=ids.device)], dim=0)
            # pad to T
            if new_seq.numel() < T:
                pad = torch.full((T - new_seq.numel(),), tokenizer.pad_token_id, device=ids.device)
                new_seq = torch.cat([new_seq, pad], dim=0)
            ids[b] = new_seq
            mask[b] = (ids[b] != tokenizer.pad_token_id).long()
        else:
            # insert before [SEP]
            cls = ids[b, 0].item()
            body = ids[b, 1:seq_len-1]
            sep  = ids[b, seq_len-1].item()
            new_body = torch.cat([body, trig], dim=0)[: (T-2)]
            new_seq = torch.tensor([cls], device=ids.device)
            new_seq = torch.cat([new_seq, new_body, torch.tensor([sep], device=ids.device)], dim=0)
            if new_seq.numel() < T:
                pad = torch.full((T - new_seq.numel(),), tokenizer.pad_token_id, device=ids.device)
                new_seq = torch.cat([new_seq, pad], dim=0)
            ids[b] = new_seq
            mask[b] = (ids[b] != tokenizer.pad_token_id).long()
    return ids, mask

# -----------------------------
# 6) HotFlip-style universal trigger mining (discrete tokens)
# -----------------------------
@dataclass
class TriggerConfig:
    k_tokens: int = 5      # trigger length
    steps: int = 30          # optimization steps
    batch_size: int = 16
    max_batches_for_mining: int = 64  # number of batches of clean data to mine over
    target_mode: str = "max_loss"     # "max_loss" or "target_label"
    target_label: int = 1             # used if target_mode == "target_label"
    place: str = "prepend"            # where to insert

TRIG_CFG = TriggerConfig()

def gather_embedding_matrix(model) -> torch.Tensor:
    return model.bert.embeddings.word_embeddings.weight  

def pick_hotflip_replacement(grad_vec: torch.Tensor, emb_matrix: torch.Tensor, banned: set, topk: int = 1) -> List[int]:
    """
    Given gradient wrt embedding at a position [d], select tokens that would maximally increase loss (dot with grad).
    We simply rank by dot(emb_i, grad). Exclude banned tokens.
    """
    # scores_i = emb_i · grad
    scores = emb_matrix @ grad_vec  
    scores = scores.detach().cpu().numpy()
    for tok in banned:
        scores[tok] = -1e9
    top_idx = np.argpartition(-scores, kth=topk-1)[:topk]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx.tolist()

def mine_universal_trigger(
    model, tokenizer, dataset: Dataset, cfg: TriggerConfig
) -> List[int]:
    model.train()  
    emb_matrix = gather_embedding_matrix(model)
    vocab_size, d = emb_matrix.shape

    # initialize trigger as random lowercase tokens not special/pad
    banned = {
        tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id,
        tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
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

        # Build labels for loss:
        if cfg.target_mode == "target_label" and model.num_labels > 1:
            y = torch.full((ids_tr.size(0),), cfg.target_label, device=device, dtype=torch.long)
            outputs = model(input_ids=ids_tr, attention_mask=mask_tr, labels=y)
            loss = outputs.loss
        else:
            logits = model(input_ids=ids_tr, attention_mask=mask_tr).logits  
            probs = torch.softmax(logits, dim=-1)
            pseudo = torch.argmax(probs, dim=-1)
            loss = nn.CrossEntropyLoss()(logits, pseudo)

        model.zero_grad(set_to_none=True)
        loss.backward()
        wgrad = model.bert.embeddings.word_embeddings.weight.grad  
        assert wgrad is not None, "No gradient captured for embeddings."

        with torch.no_grad():
            new_trig = []
            for pos, tid in enumerate(trig.tolist()):
                grad_vec = wgrad[tid]  
                # pick replacement
                choice = pick_hotflip_replacement(grad_vec, emb_matrix, banned, topk=1)[0]
                new_trig.append(choice)
            trig = torch.tensor(new_trig, device=device)

        steps_done += 1
        if steps_done >= cfg.steps:
            break

        if steps_done % 5 == 0:
            toks = tokenizer.convert_ids_to_tokens(trig.tolist())
            print(f"[Mining] step={steps_done} trigger={toks}")

        if steps_done >= cfg.max_batches_for_mining:
            break

    model.eval()
    print("Final trigger ids:", trig.tolist(), "->", tokenizer.convert_ids_to_tokens(trig.tolist()))
    return trig.tolist()

# -----------------------------
# 7) Overlap ratio measurement (A/B) per layer
# -----------------------------
@torch.no_grad()
def compute_overlap_ratios(
    model, tokenizer, dataset: Dataset, trigger_ids: List[int], place: str = "prepend",
    batch_size: int = 32, max_batches: int = 200
) -> Dict[str, List[float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    layer_counts_A = None  
    layer_counts_B = None  
    n_layers = len(model.bert.encoder.layer)
    n_seen = 0

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        # clean activations
        clean_cls = get_layer_activations_cls(input_ids, attn)       
        clean_active = boolean_active(clean_cls)

        # poisoned activations (insert trigger)
        ids_p, mask_p = insert_trigger(input_ids, attn, trigger_ids, place=place)
        poison_cls = get_layer_activations_cls(ids_p, mask_p)
        poison_active = boolean_active(poison_cls)

        # compute A/B per layer in this batch
        Bsz = input_ids.size(0)
        if layer_counts_A is None:
            layer_counts_A = [0.0] * n_layers
            layer_counts_B = [0.0] * n_layers

        for l in range(n_layers):
            ca = clean_active[l]
            pa = poison_active[l]
            A = (ca & pa).sum(dim=1).float()   
            Bc = pa.sum(dim=1).float()         
            ratio = torch.where(Bc > 0, A / Bc.clamp(min=1.0), torch.ones_like(Bc))
            layer_counts_A[l] += A.sum().item()
            layer_counts_B[l] += Bc.sum().item()

        n_seen += Bsz
        if (i + 1) >= max_batches:
            break

    ratios = []
    for l in range(n_layers):
        A_sum = layer_counts_A[l]
        B_sum = layer_counts_B[l]
        r = (A_sum / B_sum) if B_sum > 0 else 1.0
        ratios.append(float(r))
    return {"overlap_ratio_per_layer": ratios, "num_examples": n_seen}

# -----------------------------
# 8) Pipeline: mine trigger -> measure overlap
# -----------------------------
MINING_LIMIT = 20000
EVAL_LIMIT   = 200
BATCH_EVAL   = 32

full_ds   = TsvTextDataset(TSV_PATH, tokenizer, max_len=128, limit=None)
mine_ds   = TsvTextDataset(TSV_PATH, tokenizer, max_len=128, limit=MINING_LIMIT)
eval_ds   = TsvTextDataset(TSV_PATH, tokenizer, max_len=128, limit=EVAL_LIMIT)

print(f"Dataset sizes: mine={len(mine_ds)} eval={len(eval_ds)}")

trigger_ids = mine_universal_trigger(model, tokenizer, mine_ds, TRIG_CFG)
trigger_tokens = tokenizer.convert_ids_to_tokens(trigger_ids)
print("Discovered universal trigger:", trigger_tokens)

res = compute_overlap_ratios(
    model, tokenizer, eval_ds, trigger_ids,
    place=TRIG_CFG.place, batch_size=BATCH_EVAL, max_batches=math.ceil(len(eval_ds)/BATCH_EVAL)
)

print("\n=== Overlap Ratio Results ===")
for i, r in enumerate(res["overlap_ratio_per_layer"], start=1):
    print(f"Layer {i:02d}: {r:.4f}")
print(f"Aggregated over {res['num_examples']} examples.")
print("Trigger tokens:", trigger_tokens)