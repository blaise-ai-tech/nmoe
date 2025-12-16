from __future__ import annotations

import argparse
import time
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import Transformer, pool_hidden
from .docid import parse_doc_id, shard_path


DIMS = ["helpfulness", "correctness", "coherence", "complexity", "density"]


@dataclass
class TrainConfig:
    checkpoint: str
    labels_path: str
    out_dir: str
    batch_size: int = 32
    max_seq_len: int = 4096
    epochs: int = 1
    lr: float = 1e-3
    data_root: str = "."


class ShardLabelsDataset(Dataset):
    """Read token windows from shards using doc_id mapping in labels."""

    def __init__(self, labels_jsonl: str, data_root: str, max_seq_len: int = 4096):
        import numpy as np

        self.data_root = data_root
        self.max_seq_len = int(max_seq_len)
        rows: List[Dict[str, Any]] = []
        with open(labels_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                s = j.get("scores")
                doc_id = j.get("doc_id")
                if not s or not doc_id:
                    continue
                # Support both dict and 5‑element list schemas
                if isinstance(s, dict):
                    vals = [float(s.get(k, 0.0)) for k in DIMS]
                elif isinstance(s, list) and len(s) == 5:
                    vals = [float(s[i]) for i in range(5)]
                else:
                    continue
                target = torch.tensor(vals, dtype=torch.float32)
                rows.append({"doc_id": doc_id, "target": target})
        if not rows:
            raise ValueError("no valid rows in labels_jsonl; expected {doc_id,scores}")
        self.rows = rows
        self._np = np

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        doc = parse_doc_id(row["doc_id"])
        p = shard_path(self.data_root, doc)
        arr = self._np.load(p, mmap_mode="r")
        start, end = int(doc.start), int(doc.end)
        toks = arr[start:end].astype(self._np.int64)
        if len(toks) > self.max_seq_len:
            toks = toks[: self.max_seq_len]
        return {"input_ids": torch.from_numpy(toks.copy()).long(), "target": row["target"], "doc_id": row["doc_id"]}


class ProbeHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 5)

    def forward(self, h18: torch.Tensor) -> torch.Tensor:
        return self.proj(h18)


class MTPJudgeHead(nn.Module):
    """MTP-style Judge head (depth=1) for grading.

    - Prepend a learned quality token to L24 sequence
    - Project to mid_dim, pass through a tiny encoder (depth 0)
    - Predict 5 dims via a shared head from the quality token readout
    - Depth 1: inject an embedding of (teacher) scores at the quality token,
      pass through another tiny encoder, and predict again with the same head.
    """

    def __init__(self, hidden_dim: int, mid_dim: int = 512, nhead: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mid_dim = mid_dim
        self.q0 = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.q0, mean=0.0, std=0.02)
        self.proj = nn.Linear(hidden_dim, mid_dim)
        self.enc0 = nn.TransformerEncoderLayer(d_model=mid_dim, nhead=nhead, batch_first=True)
        self.shared_head = nn.Linear(mid_dim, 5)
        self.score_embed = nn.Sequential(
            nn.Linear(5, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        self.enc1 = nn.TransformerEncoderLayer(d_model=mid_dim, nhead=nhead, batch_first=True)

    def forward(self, h24_seq: torch.Tensor, teacher_scores: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args:
            h24_seq: [B, T, H] hidden states from layer 24
            teacher_scores: [B, 5] float labels for teacher forcing (optional)

        Returns:
            (scores_depth0 [B,5], scores_depth1 [B,5]|None)
        """
        B, T, H = h24_seq.shape
        q = self.q0.unsqueeze(0).unsqueeze(1).expand(B, 1, H)  # [B,1,H]
        seq = torch.cat([q, h24_seq], dim=1)  # [B,T+1,H]
        z0 = self.proj(seq)  # [B,T+1,mid]
        z0 = self.enc0(z0)
        q0_out = z0[:, 0, :]  # [B,mid]
        s0 = self.shared_head(q0_out)

        s1 = None
        if teacher_scores is not None:
            # Normalize teacher to [0,1] and embed
            t = torch.clamp(teacher_scores.float() / 4.0, 0.0, 1.0)
            t_emb = self.score_embed(t)  # [B,mid]
            z1 = z0.clone()
            z1[:, 0, :] = z1[:, 0, :] + t_emb
            z1 = self.enc1(z1)
            q1_out = z1[:, 0, :]
            s1 = self.shared_head(q1_out)
        return s0, s1


def _collate_rows(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(len(b["input_ids"]) for b in batch)
    bsz = len(batch)
    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    positions = torch.zeros(bsz, max_len, dtype=torch.long)
    targets = torch.stack([b["target"] for b in batch])
    for i, b in enumerate(batch):
        n = len(b["input_ids"]) 
        input_ids[i, :n] = b["input_ids"]
        positions[i, :n] = torch.arange(n, dtype=torch.long)
    doc_ids = [b["doc_id"] for b in batch]
    return {"input_ids": input_ids, "positions": positions, "target": targets, "doc_ids": doc_ids}


def train_probe(cfg: TrainConfig) -> None:
    device = torch.device("cuda")
    ds = ShardLabelsDataset(cfg.labels_path, data_root=cfg.data_root, max_seq_len=cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, collate_fn=_collate_rows)

    model = Transformer.from_checkpoint(cfg.checkpoint, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    head = ProbeHead(model.config.hidden_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    head.train()
    step = 0
    log_every = int(os.environ.get("HYDRA_LOG_EVERY", "50"))
    for _ in range(cfg.epochs):
        for batch in dl:
            t0 = time.perf_counter()
            input_ids = batch["input_ids"].to(device)
            positions = batch["positions"].to(device)
            targets = batch["target"].to(device)
            with torch.no_grad():
                _, h = model(
                    input_ids,
                    positions,
                    return_hidden_states=True,
                    up_to_layer=model.config.num_hidden_layers,
                    no_logits=True,
                )
                h18 = h.get(18)
                if h18 is None:
                    raise RuntimeError("Layer 18 hidden state not captured")
                pooled = pool_hidden(h18).float()
            preds = head(pooled)
            loss = loss_fn(preds, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step == 1 or (step % log_every == 0):
                dt = (time.perf_counter() - t0) * 1000.0
                bsz = int(input_ids.size(0))
                mem_gb = float(torch.cuda.memory_allocated(device) / (1024 ** 3)) if torch.cuda.is_available() else 0.0
                print(f"[hydra][probe] step={step} loss={loss.item():.4f} ms/b={dt:.1f} rows/s={bsz*1000.0/dt:.1f} mem_gb={mem_gb:.2f}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out_dir / "hydra_probe.pt")


def train_judge(cfg: TrainConfig) -> None:
    device = torch.device("cuda")
    ds = ShardLabelsDataset(cfg.labels_path, data_root=cfg.data_root, max_seq_len=cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, collate_fn=_collate_rows)

    model = Transformer.from_checkpoint(cfg.checkpoint, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    head = MTPJudgeHead(model.config.hidden_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    ln = nn.LayerNorm(model.config.hidden_size).to(device)

    head.train()
    step = 0
    log_every = int(os.environ.get("HYDRA_LOG_EVERY", "50"))
    warmup_steps = int(os.environ.get("HYDRA_WARMUP_STEPS", "500"))
    nan_steps = 0
    for _ in range(cfg.epochs):
        for batch in dl:
            t0 = time.perf_counter()
            input_ids = batch["input_ids"].to(device)
            positions = batch["positions"].to(device)
            targets = batch["target"].to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                    _, h = model(
                        input_ids,
                        positions,
                        return_hidden_states=True,
                        up_to_layer=model.config.num_hidden_layers,
                        no_logits=True,
                    )
                h24 = h.get(24)
                if h24 is None:
                    raise RuntimeError("Layer 24 hidden state not captured")
            # Stabilize hidden states before the Judge head
            h24f = torch.nan_to_num(h24, nan=0.0, posinf=1e4, neginf=-1e4).to(torch.float32)
            h24f = ln(h24f)
            s0, s1 = head(h24f, teacher_scores=targets)
            # Clamp to label domain [0,4] for loss computation only
            tgt = torch.clamp(targets, 0.0, 4.0)
            p0 = torch.clamp(s0, 0.0, 4.0)
            loss0 = loss_fn(p0, tgt)
            if s1 is not None:
                p1 = torch.clamp(s1, 0.0, 4.0)
                loss1 = loss_fn(p1, tgt)
            else:
                loss1 = torch.tensor(0.0, device=device)
            loss = loss0 + 0.5 * loss1
            if not torch.isfinite(loss):
                nan_steps += 1
                # Skip step to avoid corrupting optimizer state
                if step == 0 or (nan_steps % max(1, log_every) == 0):
                    print(f"[hydra][judge] step={step} loss=nan (skipped) nan_steps={nan_steps}")
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping + warmup scaling
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
            if warmup_steps > 0:
                lr_scale = min(1.0, step / float(warmup_steps))
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * lr_scale
            opt.step()
            step += 1
            if step == 1 or (step % log_every == 0):
                dt = (time.perf_counter() - t0) * 1000.0
                bsz = int(input_ids.size(0))
                mem_gb = float(torch.cuda.memory_allocated(device) / (1024 ** 3)) if torch.cuda.is_available() else 0.0
                current_lr = opt.param_groups[0]["lr"] if opt.param_groups else cfg.lr
                print(
                    f"[hydra][judge] step={step} loss={loss.item():.4f} ms/b={dt:.1f} rows/s={bsz*1000.0/dt:.1f} "
                    f"mem_gb={mem_gb:.2f} lr={current_lr:.2e} grad_norm={grad_norm:.2f} nan_steps={nan_steps}"
                )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out_dir / "hydra_judge.pt")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("nmoe.data.train")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_probe = sp.add_parser("probe", help="Train HYDRA Probe head on 120B labels")
    p_probe.add_argument("--labels", required=True)
    p_probe.add_argument("--checkpoint", required=True)
    p_probe.add_argument("--out", required=True)
    p_probe.add_argument("--batch-size", type=int, default=32)
    p_probe.add_argument("--epochs", type=int, default=1)
    p_probe.add_argument("--lr", type=float, default=1e-3)
    p_probe.add_argument("--data-root", required=True)
    p_probe.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=1024)

    p_judge = sp.add_parser("judge", help="Train HYDRA Judge head on 120B labels")
    p_judge.add_argument("--labels", required=True)
    p_judge.add_argument("--checkpoint", required=True)
    p_judge.add_argument("--out", required=True)
    p_judge.add_argument("--batch-size", type=int, default=16)
    p_judge.add_argument("--epochs", type=int, default=1)
    p_judge.add_argument("--lr", type=float, default=1e-4)
    p_judge.add_argument("--data-root", required=True)
    p_judge.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=1024)

    return p


def main(argv: List[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.cmd == "probe":
        cfg = TrainConfig(
            checkpoint=args.checkpoint,
            labels_path=args.labels,
            out_dir=args.out,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            data_root=args.data_root,
        )
        train_probe(cfg)
        return 0
    elif args.cmd == "judge":
        cfg = TrainConfig(
            checkpoint=args.checkpoint,
            labels_path=args.labels,
            out_dir=args.out,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            data_root=args.data_root,
        )
        train_judge(cfg)
        return 0
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
