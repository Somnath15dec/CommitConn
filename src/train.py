"""Step F: LoRA fine-tune of Qwen/Qwen3-0.6B to generate Semantic IDs (code only until the GPU is free).

- New tokens <a_0..255> <b_0..255> <c_0..255> <d_0..max_d>; embeddings resized (mean-initialized by transformers).
- LoRA r=16, alpha=32, dropout 0.05 on q,k,v,o,gate,up,down. The NEW token rows are trainable through PEFT
  `trainable_token_indices` (peft >= 0.15). Qwen3-0.6B ties embed_tokens and lm_head; PEFT applies the same
  trainable-token delta to the tied lm_head, so only the 777 new rows train instead of the full 152k x 1024 matrix
  that modules_to_save would copy (2 x 156M params).
- Example: "Query: {q}\\nSID:" + 4 SID tokens + eos, left-padded. Loss on the 4 SID tokens only: logits are kept
  for the last 6 positions only (logits_to_keep), so the 152k-way softmax runs on 4 positions, not 64.
- bf16 autocast (adapters fp32), max_len 64, micro-batch 16 x accum 4 = 64, AdamW lr 2e-4, cosine, 3 epochs,
  checkpoint per epoch in checkpoints/. Per epoch: train loss, val loss, val SID-valid rate (greedy, unconstrained),
  val exact-track Hit@10 (constrained beam 20), dev playlist precision@10 (src/dev_split.py, constrained beam 20
  over the dev pool). Train / val pairs come from the frozen files (python -m src.train_data --freeze).

    python -m src.train --run run1   # full run (GPU): checkpoints/run1/epoch{1..3}, train_log.jsonl, summary.json
    python -m src.train --smoke      # 500 pairs, 50 steps

summary.json: per-epoch table; best epoch = highest dev (b) precision@10, ties broken by lower val loss.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.decode import LEVELS, PROMPT, SidDecoder, sid_token, sid_token_list
from src.paths import ROOT
from src.dev_split import K_CONDENSED, dev_metrics, dev_tracks, training_tune_tracks
from src.train_data import SIDS, epoch_indices, frozen_split

BASE_MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINTS = ROOT / "checkpoints"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 64
MICRO_BATCH, GRAD_ACCUM = 16, 4          # effective batch 64
LR, EPOCHS = 2e-4, 3
WARMUP_STEPS = 30                        # linear warmup, then cosine decay
VAL_BEAMS, VAL_K = 20, 10
DECODE_BATCH_20, DECODE_BATCH_50 = 8, 2     # queries per generate() call; keeps beam-search logits inside 8 GB
SMOKE_PAIRS, SMOKE_STEPS, SMOKE_VAL = 500, 50, 100
SEED = 20
N_SID = len(LEVELS)
KEEP = N_SID + 2                         # logits for the last 6 positions: those at -6..-3 predict the 4 SID tokens


# ---------------------------------------------------------------- tokens / examples
def add_sid_tokens(tokenizer, sids: pd.DataFrame) -> list[int]:
    toks = sid_token_list(sids)
    tokenizer.add_tokens(toks, special_tokens=True)
    return tokenizer.convert_tokens_to_ids(toks)


def sid_ids(tokenizer, sid) -> list[int]:
    return tokenizer.convert_tokens_to_ids([sid_token(lv, int(v)) for lv, v in zip(LEVELS, sid)])


def encode(tokenizer, query: str, sid) -> tuple[list[int], list[int]]:
    """(input_ids, labels): prompt + 4 SID tokens + eos; labels -100 everywhere except the SID tokens.
    The prompt is truncated from the left of the query so the sequence fits MAX_LEN."""
    tail = sid_ids(tokenizer, sid) + [tokenizer.eos_token_id]
    prompt = tokenizer(PROMPT.format(q=query), add_special_tokens=False).input_ids
    prompt = prompt[-(MAX_LEN - len(tail)):]
    ids = prompt + tail
    labels = [-100] * len(prompt) + tail[:N_SID] + [-100]
    return ids, labels


def collate(examples: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """Left padding, so the SID tokens always sit at positions -5..-2 (eos at -1)."""
    L = max(len(i) for i, _ in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for r, (i, l) in enumerate(examples):
        ids[r, L - len(i):], lab[r, L - len(l):], att[r, L - len(i):] = torch.tensor(i), torch.tensor(l), 1
    pos = (att.cumsum(1) - 1).clamp(min=0)
    return {"input_ids": ids, "attention_mask": att, "position_ids": pos, "labels": lab}


def sid_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"], logits_to_keep=KEEP)
    logits = out.logits[:, :N_SID].float()                       # positions -6..-3
    target = batch["labels"][:, -(N_SID + 1):-1]                 # SID tokens at -5..-2
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


# ---------------------------------------------------------------- model
def load_tokenizer(name: str = BASE_MODEL):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    return tok


def wrap_lora(model, new_ids: list[int]):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS,
                     trainable_token_indices={"embed_tokens": new_ids}, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def build_model(tokenizer, sids: pd.DataFrame, base=None, name: str = BASE_MODEL):
    """(peft model, new token ids). `base` lets tests pass a tiny random model."""
    new_ids = add_sid_tokens(tokenizer, sids)
    if base is None:
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    base.resize_token_embeddings(len(tokenizer))
    return wrap_lora(base, new_ids), new_ids


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def val_loss(model, tok, val: pd.DataFrame, device) -> float:
    model.eval()
    losses = []
    rows = list(val[["query", *LEVELS]].itertuples(index=False))
    for s in range(0, len(rows), MICRO_BATCH * 2):
        b = collate([encode(tok, r[0], r[1:]) for r in rows[s:s + MICRO_BATCH * 2]], tok.pad_token_id)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses.append(sid_loss(model, {k: v.to(device) for k, v in b.items()}).item() * len(b["input_ids"]))
    return float(np.sum(losses) / len(rows))


@torch.no_grad()
def sid_valid_rate(model, tok, dec: SidDecoder, queries: list[str], batch: int = 64) -> float:
    """Greedy, unconstrained: share of generations whose first 4 new tokens are a catalog SID."""
    model.eval()
    ok = 0
    for s in range(0, len(queries), batch):
        enc = tok([PROMPT.format(q=q) for q in queries[s:s + batch]], return_tensors="pt", padding=True).to(
            model.device)
        g = model.generate(**enc, max_new_tokens=N_SID, do_sample=False, pad_token_id=tok.pad_token_id)
        ok += sum(dec.lookup(r) is not None for r in g[:, enc.input_ids.shape[1]:].tolist())
    return ok / len(queries)


def hit_at_k(dec: SidDecoder, val: pd.DataFrame, k: int = VAL_K, beams: int = VAL_BEAMS) -> float:
    res = dec.search(val["query"].tolist(), n=k, num_beams=beams, batch_size=DECODE_BATCH_20)
    return float(np.mean([t in {x for x, _ in r} for t, r in zip(val.track_id, res)]))


# ---------------------------------------------------------------- training
def cosine_lr(step: int, total: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def best_epoch(table: list[dict]) -> dict:
    """Highest dev (b) precision@10; ties: lower val loss."""
    return max(table, key=lambda r: (r["dev_b_precision@10"], -r["val_loss"]))


def train(smoke: bool = False, run_name: str = "run1") -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sids = pd.read_parquet(SIDS)
    tr, val = frozen_split()
    if smoke:
        tr = tr.sample(SMOKE_PAIRS, random_state=SEED).reset_index(drop=True)
        val = val.head(SMOKE_VAL)
    tok = load_tokenizer()
    model, new_ids = build_model(tok, sids)
    model.to(device)
    trainable, total = count_params(model)
    print(f"train {tr.source.value_counts().to_dict()} | val {len(val):,} | new tokens {len(new_ids)} | "
          f"trainable {trainable:,} / {total:,}", flush=True)
    dec = SidDecoder(model, tok, sids)
    dev_dec_b = SidDecoder(model, tok, sids, exclude=set(sids.track_id) - dev_tracks())     # within dev tracks
    dev_dec_a = SidDecoder(model, tok, sids, exclude=training_tune_tracks())                # dev pool
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    epochs = 1 if smoke else EPOCHS
    steps_per_epoch = [math.ceil(len(epoch_indices(tr, e)) / (MICRO_BATCH * GRAD_ACCUM)) for e in range(epochs)]
    total_steps = SMOKE_STEPS if smoke else sum(steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total_steps))
    rows = tr[["query", *LEVELS]].to_numpy(dtype=object)
    run_dir = CHECKPOINTS / ("smoke" if smoke else run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    table, per_word = [], {}
    step = 0

    def opt_step():
        nonlocal step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1

    for epoch in range(epochs):
        idx = epoch_indices(tr, epoch)
        if smoke:
            idx = np.resize(idx, SMOKE_STEPS * MICRO_BATCH * GRAD_ACCUM)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, run, n, group, t_step = time.time(), 0.0, 0, 0.0, time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for s in range(0, len(idx), MICRO_BATCH):
            b = collate([encode(tok, r[0], r[1:]) for r in rows[idx[s:s + MICRO_BATCH]]], tok.pad_token_id)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = sid_loss(model, {k: v.to(device) for k, v in b.items()})
            (loss / GRAD_ACCUM).backward()
            run, n, group = run + loss.item(), n + 1, group + loss.item()
            if n % GRAD_ACCUM == 0:
                opt_step()
                if smoke or step % 50 == 0:
                    rec_s = {"step": step, "loss": group / GRAD_ACCUM, "lr": sched.get_last_lr()[0],
                             "sec_per_step": time.time() - t_step}
                    print(json.dumps(rec_s), flush=True)
                group, t_step = 0.0, time.time()
                if step >= total_steps:
                    break
        if n % GRAD_ACCUM and step < total_steps:     # flush the last partial accumulation group of the epoch
            opt_step()
        train_secs = time.time() - t0
        peak_train = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
        rec = {"epoch": epoch + 1, "step": step, "train_loss": run / max(1, n),
               "val_loss": val_loss(model, tok, val, device),
               "val_sid_valid_greedy": sid_valid_rate(model, tok, dec, val["query"].tolist()),
               "val_hit@10_constrained": hit_at_k(dec, val), "train_seconds": train_secs}
        dm = dev_metrics(
            lambda qs: [[t for t, _ in r] for r in dev_dec_b.search(qs, n=VAL_K, num_beams=VAL_BEAMS, batch_size=DECODE_BATCH_20)],
            lambda qs: [[t for t, _ in r] for r in dev_dec_a.search(qs, n=K_CONDENSED, num_beams=K_CONDENSED, batch_size=DECODE_BATCH_50)])
        rec.update({k: v for k, v in dm.items() if k not in ("a", "b")})
        rec["epoch_seconds"] = time.time() - t0
        if device.type == "cuda":
            rec["peak_vram_gb_train"] = peak_train
            rec["peak_vram_gb_with_eval"] = torch.cuda.max_memory_allocated() / 1e9
        table.append(rec)
        per_word[epoch + 1] = {"b": dm["b"]["per_word"], "a": dm["a"]["per_word"]}
        print(json.dumps(rec), flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if not smoke:
            out = run_dir / f"epoch{epoch + 1}"
            model.save_pretrained(out)
            tok.save_pretrained(out)
    best = best_epoch(table)
    summary = {"run": run_dir.name, "seed": SEED, "epochs": table, "best_epoch": best["epoch"],
               "best_rule": "highest dev_b_precision@10; ties: lower val_loss", "dev_per_word": per_word,
               "config": {"lr": LR, "warmup_steps": WARMUP_STEPS, "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
                          "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
                          "max_len": MAX_LEN, "total_steps": total_steps, "train_pairs": len(tr), "val_pairs": len(val)}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"best epoch: {best['epoch']} (dev b {best['dev_b_precision@10']:.3f}, val loss {best['val_loss']:.4f})",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", default="run1")
    a = ap.parse_args()
    train(a.smoke, a.run)
    sys.exit(0)
