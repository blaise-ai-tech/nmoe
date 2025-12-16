from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from nmoe.metrics import start_metrics, stop_metrics
from nmoe.eval.adapters import iter_choices, iter_span


# -----------------------------
# Utilities
# -----------------------------

def _load_snapshot(path: Path) -> tuple[dict, dict]:
    obj = torch.load(str(path), map_location="cpu")
    cfg_dict = obj.get("config", {})
    model_state = obj.get("model_state", {})
    return cfg_dict, model_state


def _hash_u32(s: str, seed: int = 0) -> int:
    h = hashlib.blake2s((str(seed) + s).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


# -----------------------------
# Simulation model (deterministic)
# -----------------------------

@dataclass
class SimConfig:
    acc_choices: float = 0.6  # target accuracy for choices tasks
    acc_span_em: float = 0.55  # exact-match rate for span tasks
    acc_unittest: float = 0.5  # pass rate
    acc_judge: float = 0.6     # keep rate
    seed: int = 123


class SimModel:
    """Deterministic pseudo-model for end-to-end eval plumbing tests.

    Produces controlled accuracies without requiring a trained model.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def predict_choice(self, prompt: str, n_choices: int, correct_idx: int) -> int:
        # Hash prompt → pseudo RNG; hit target accuracy by thresholding
        r = (_hash_u32(prompt, self.cfg.seed) % 10000) / 10000.0
        if r < self.cfg.acc_choices:
            return int(correct_idx)
        else:
            # Pick a wrong answer deterministically
            return (correct_idx + 1 + (int(r * 17) % max(1, n_choices - 1))) % n_choices

    def generate_span(self, prompt: str, answer: str) -> str:
        r = (_hash_u32(prompt + answer, self.cfg.seed) % 10000) / 10000.0
        if r < self.cfg.acc_span_em:
            return answer
        # Return a near-miss string
        return answer.lower().strip().rstrip(".") + " maybe"

    def run_unittest(self, prompt: str) -> bool:
        r = (_hash_u32("UT:" + prompt, self.cfg.seed) % 10000) / 10000.0
        return r < self.cfg.acc_unittest

    def judge_keep(self, prompt: str) -> bool:
        r = (_hash_u32("J:" + prompt, self.cfg.seed) % 10000) / 10000.0
        return r < self.cfg.acc_judge


# -----------------------------
# Task loading (TOML)
# -----------------------------

def _load_tasks_toml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        import tomllib
        with path.open("rb") as f:
            obj = tomllib.load(f)
        tasks = obj.get("task", [])
        return tasks if isinstance(tasks, list) else []
    except Exception:
        return []


# -----------------------------
# Example samplers (simulation)
# -----------------------------

def _sim_examples_choices(n: int, n_choices: int = 4) -> list[dict]:
    out = []
    for i in range(n):
        prompt = f"Q{i}: What is the best option?"
        choices = [f"Option {chr(65+j)}" for j in range(n_choices)]
        correct = i % n_choices
        out.append({"prompt": prompt, "choices": choices, "label": correct})
    return out


def _sim_examples_span(n: int) -> list[dict]:
    out = []
    for i in range(n):
        prompt = f"S{i}: The capital of France is"
        answer = "Paris"
        out.append({"prompt": prompt, "answers": [answer]})
    return out


def _sim_examples_unittest(n: int) -> list[dict]:
    out = []
    for i in range(n):
        prompt = f"Write a function f{i}(x) that returns x+1"
        out.append({"prompt": prompt, "tests": ["assert f(1)==2", "assert f(5)==6"]})
    return out


def _sim_examples_judge(n: int) -> list[dict]:
    out = []
    for i in range(n):
        prompt = f"Explain backpropagation in 2 sentences (item {i})"
        out.append({"prompt": prompt})
    return out


# -----------------------------
# Scorers
# -----------------------------

def _centered(acc: float, baseline: float) -> float:
    # (acc - b) / (1 - b), clipped to [0,1] for stability in simulation
    if baseline >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (acc - baseline) / (1.0 - baseline)))


BASELINE_BY_TASK: Dict[str, float] = {
    # Random baselines (approx). For public parity, swap with DCLM v2 values later.
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "HellaSwag": 0.25,
    "MMLU": 0.25,
    "OpenBookQA": 0.25,
    "PIQA": 0.5,         # 2-choice
    "SIQA": 0.3333,      # 3-choice
    "WinoGrande": 0.5,   # binary
    "BoolQ": 0.5,
    "COPA": 0.5,         # 2-choice
    "StoryCloze": 0.5,   # 2-choice
    "SQuAD v1.1": 0.0,   # EM baseline treated as 0
}


@dataclass
class EvalResult:
    task: str
    raw_acc: float
    centered_acc: float
    n: int


def _run_choices_sim(task_name: str, model: SimModel, n: int) -> EvalResult:
    # Determine number of choices from task family
    n_choices = 4
    if task_name in ("PIQA", "BoolQ", "COPA", "StoryCloze", "WinoGrande"):
        n_choices = 2
    if task_name == "SIQA":
        n_choices = 3
    ex = _sim_examples_choices(n, n_choices)
    correct = 0
    for e in ex:
        pred = model.predict_choice(e["prompt"], len(e["choices"]), e["label"])
        correct += int(pred == int(e["label"]))
    acc = correct / max(1, n)
    centered = _centered(acc, BASELINE_BY_TASK.get(task_name, 0.0))
    return EvalResult(task=task_name, raw_acc=acc, centered_acc=centered, n=n)


def _normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().replace("\n", " ").split())


def _run_span_sim(task_name: str, model: SimModel, n: int) -> EvalResult:
    ex = _sim_examples_span(n)
    em = 0
    for e in ex:
        pred = model.generate_span(e["prompt"], e["answers"][0])
        if _normalize_text(pred) in {_normalize_text(a) for a in e["answers"]}:
            em += 1
    acc = em / max(1, n)
    centered = _centered(acc, BASELINE_BY_TASK.get(task_name, 0.0))
    return EvalResult(task=task_name, raw_acc=acc, centered_acc=centered, n=n)


def _run_unittest_sim(task_name: str, model: SimModel, n: int) -> EvalResult:
    ex = _sim_examples_unittest(n)
    passed = 0
    for e in ex:
        if model.run_unittest(e["prompt"]):
            passed += 1
    acc = passed / max(1, n)
    centered = _centered(acc, 0.0)
    return EvalResult(task=task_name, raw_acc=acc, centered_acc=centered, n=n)


def _run_judge_sim(task_name: str, model: SimModel, n: int, tau_keep: float = 0.5) -> EvalResult:
    ex = _sim_examples_judge(n)
    kept = 0
    for e in ex:
        if model.judge_keep(e["prompt"]):
            kept += 1
    acc = kept / max(1, n)
    centered = _centered(acc, tau_keep)  # treat threshold as baseline for centering
    return EvalResult(task=task_name, raw_acc=acc, centered_acc=centered, n=n)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser("nmoe.eval.runner")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tasks", default="core")
    ap.add_argument("--max-examples", type=int, default=500)
    ap.add_argument("--max-time", type=int, default=300)
    ap.add_argument("--tasks-file", default="configs/eval/tasks.toml")
    # Simulation controls (for pre-training validation)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-acc-choices", type=float, default=0.6)
    ap.add_argument("--sim-acc-span", type=float, default=0.55)
    ap.add_argument("--sim-acc-unittest", type=float, default=0.5)
    ap.add_argument("--sim-acc-judge", type=float, default=0.6)
    ap.add_argument("--sim-seed", type=int, default=123)
    args = ap.parse_args()

    snap = Path(args.snapshot)
    cfg_dict, _model_state = _load_snapshot(snap)

    # Metrics context (use same run id if present)
    run_id = cfg_dict.get("experiment_id", "eval")
    ctx = start_metrics(run_id=run_id, metrics_dir=cfg_dict.get("metrics_dir", "/data/metrics"))

    # Load task specs
    tasks_spec = _load_tasks_toml(Path(args.tasks_file))
    # Map name->spec for quick lookup
    spec_by_name = {t.get("name"): t for t in tasks_spec if isinstance(t, dict) and t.get("name")}

    # Build the real model (GPU) for scoring
    from nmoe.config import Config
    from nmoe.model import Transformer
    cfg = Config(**cfg_dict)
    model = Transformer(cfg).cuda().eval()
    # Load state if present
    try:
        from torch.nn.modules.module import _IncompatibleKeys
        miss = model.load_state_dict(_model_state, strict=False)
    except Exception:
        pass

    t0 = _now_ms()
    results: list[EvalResult] = []

    # Determine task list
    if args.tasks == "core":
        # Prefer tasks file ordering if available
        if tasks_spec:
            task_names = [t.get("name") for t in tasks_spec if isinstance(t, dict) and t.get("name")]
        else:
            task_names = []
    elif args.tasks.startswith("chat"):
        parts = args.tasks.split(":", 1)
        task_names = parts[1].split("|") if len(parts) == 2 else ["ARC-Easy","ARC-Challenge","MMLU","GSM8K","HumanEval"]
    else:
        task_names = [args.tasks]

    # Tokenizer (for prompt encoding); keep minimal deps
    import tiktoken
    enc = tiktoken.get_encoding(cfg.tokenizer)

    # Run scorers per task
    for name in task_names:
        spec = spec_by_name.get(name, {})
        src = spec.get("source")
        if not src:
            # Skip tasks that are not present in tasks file
            continue
        scorer = str(spec.get("scorer", "choices")).lower()
        n = max(1, min(int(args.max_examples), 500))  # cap for simulation speed
        if scorer == "choices":
            # Batched option log-likelihood; choose argmax
            correct = 0
            total = 0
            items = list(iter_choices(name, src, n))
            for it in items:
                prompt = it["prompt"]
                options = it["options"]
                gold = int(it["label"]) if it.get("label") is not None else 0
                # Score each option continuation
                scores = []
                with torch.inference_mode():
                    for opt in options:
                        text = prompt + " " + str(opt)
                        tokens = enc.encode(text)
                        toks = torch.tensor(tokens, device="cuda", dtype=torch.long)[None, :]
                        logits = model(toks)  # [1, T, V]
                        # Compute simple next-token LL over the option span only
                        # Split prompt and option by lengths
                        p_ids = enc.encode(prompt)
                        opt_ids = enc.encode(" " + str(opt))
                        if len(opt_ids) == 0:
                            scores.append(float("-inf"))
                            continue
                        start = len(p_ids) - 1
                        # Sum log-probs across the option tokens
                        ll = 0.0
                        for i, tok in enumerate(opt_ids):
                            pos = start + i
                            if pos < 0 or pos >= logits.size(1):
                                break
                            lp = torch.log_softmax(logits[0, pos, :], dim=-1)
                            ll += float(lp[tok].item())
                        scores.append(ll)
                pred = int(max(range(len(options)), key=lambda j: scores[j])) if scores else 0
                correct += int(pred == gold)
                total += 1
            acc = correct / max(1, total)
            baseline = BASELINE_BY_TASK.get(name, 0.0)
            results.append(EvalResult(task=name, raw_acc=acc, centered_acc=_centered(acc, baseline), n=total))

        elif scorer == "span":
            # Greedy decode and EM against provided answers
            em = 0
            total = 0
            items = list(iter_span(name, spec.get("source", ""), n))
            for it in items:
                prompt = it["prompt"]; answers = it.get("answers", [])
                # Greedy one-pass decode for a small budget (no KV cache)
                toks = torch.tensor(enc.encode(prompt), device="cuda", dtype=torch.long)[None, :]
                with torch.inference_mode():
                    logits = model(toks)
                    next_id = int(torch.argmax(logits[0, -1]).item())
                out = enc.decode([next_id])
                if _normalize_text(out) in {_normalize_text(a) for a in answers}:
                    em += 1
                total += 1
            acc = em / max(1, total)
            baseline = BASELINE_BY_TASK.get(name, 0.0)
            results.append(EvalResult(task=name, raw_acc=acc, centered_acc=_centered(acc, baseline), n=total))

        elif scorer == "unittest":
            # Placeholder until sandbox harness is added (count as 0)
            results.append(EvalResult(task=name, raw_acc=0.0, centered_acc=0.0, n=0))

        elif scorer == "judge":
            # Placeholder until HYDRA judge harness is added (count as 0)
            results.append(EvalResult(task=name, raw_acc=0.0, centered_acc=0.0, n=0))
        else:
            results.append(EvalResult(task=name, raw_acc=0.0, centered_acc=0.0, n=0))

    # Aggregate CORE (average centered accuracy across tasks present)
    core = sum(r.centered_acc for r in results) / max(1, len(results))

    # Write summary JSON next to snapshot
    summary = {
        "suite": "core" if args.tasks == "core" else "custom",
        "CORE": core,
        "tasks": [
            {"task": r.task, "raw_acc": r.raw_acc, "centered_acc": r.centered_acc, "n": r.n}
            for r in results
        ],
        "elapsed_ms": _now_ms() - t0,
    }
    with (snap.parent / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Persist to metrics DB
    try:
        if ctx is not None and ctx.writer is not None:
            step = int(cfg_dict.get("global_step", 0))
            # Per-task
            items = []
            for r in results:
                items.append((f"eval/task/{r.task}/acc", float(r.raw_acc)))
                items.append((f"eval/task/{r.task}/centered", float(r.centered_acc)))
                items.append((f"eval/task/{r.task}/n", float(r.n)))
            # CORE aggregate
            items.append(("eval/CORE", float(core)))
            ctx.writer.insert_many(step, items)
    except Exception:
        pass

    stop_metrics(ctx)


if __name__ == "__main__":
    main()
