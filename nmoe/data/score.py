"""
Unified Harmony-based scoring utilities for gpt-oss models.

Design goals (production-ready, minimal):
- Harmony StreamableParser with final-channel only grading.
- Right-trim prompts to preserve the assistant-start suffix.
- Robust stop: finish only when the previous parser channel was "final"
  and the current token is an assistant-action stop.
- EOS salvage: process_eos() to finalize lingering sequences.
- Batched inference via nmoe.data.model.BatchedGenerator.

This module exposes the following helpers:
    build_prompt(enc, text, is_code=False) -> List[int]
    right_trim(prompt_ids, max_ctx=4096, max_new=2048) -> List[int]
    grade_prompts(gen, enc, prompts, max_new=2048, stop_tokens=None) -> List[dict]
    compute_aggregated(scores: dict[str, float]) -> float
    compute_aggregated_code(scores: dict[str, float]) -> float

Notes
- Schema: by default we expect JSON with five 0–4 dimensions for general grading:
  helpfulness, correctness, coherence, complexity, density.
- For code grading, CODE_RUBRIC uses 0–10 for readability/modularity/clarity/reusability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import json
import torch

from openai_harmony import (
    Conversation,
    Message,
    Role,
    StreamableParser,
)


# ============================================================================
# Rubrics (mirrors tested /tmp oracle script)
# ============================================================================

GENERAL_RUBRIC = (
    "You are a data-quality oracle for LLM pretraining. Think privately and output only strict JSON in the final channel.\n"
    "Rate this text on five dimensions, 0-4 integers each: helpfulness, correctness, coherence, complexity, density.\n"
    "Hidden analysis (not in final): (1) analyze structure; (2) identify issues; (3) assess depth; (4) then score.\n"
    "Final JSON only: {\"helpfulness\": X, \"correctness\": X, \"coherence\": X, \"complexity\": X, \"density\": X}\n"
)

CODE_RUBRIC = (
    "You are a code-quality oracle for LLM pretraining. Think privately and output only strict JSON in the final channel.\n"
    "Rate on 0-10: readability, modularity, clarity, reusability. Add zero_score_reason when applicable\n"
    "(auto_generated|data_or_config|binary_blob|duplicate|compile_error|security_risk|other).\n"
    "Hidden steps: (1) analyze structure; (2) find logic errors; (3) assess readability; (4) then score.\n"
    "Final JSON only: {\"readability\": X, \"modularity\": X, \"clarity\": X, \"reusability\": X, \"zero_score_reason\": null|\"...\"}\n"
)


# ============================================================================
# Prompt construction & trimming
# ============================================================================

def build_prompt(enc, text: str, is_code: bool = False) -> List[int]:
    """Build Harmony chat prompt tokens for ASSISTANT completion.

    The returned token list ends with the assistant-start suffix; callers
    should pass it through right_trim(...) before submission to respect context.
    """
    rubric = CODE_RUBRIC if is_code else GENERAL_RUBRIC
    conv = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, rubric),
            Message.from_role_and_content(Role.USER, f"Text:\n{text}\n\nOutput JSON only."),
        ]
    )
    return list(enc.render_conversation_for_completion(conv, Role.ASSISTANT))


def right_trim(prompt_ids: List[int], max_ctx: int = 4096, max_new: int = 2048) -> List[int]:
    """Right-trim tokens to keep the last (max_ctx - max_new) tokens.

    This preserves Harmony's assistant-start suffix at the end of the prompt.
    """
    budget = max(0, max_ctx - max_new)
    if len(prompt_ids) > budget:
        return prompt_ids[-budget:]
    return prompt_ids


# ============================================================================
# Batched streaming & parsing
# ============================================================================

@dataclass
class _SeqState:
    parser: StreamableParser
    final_buf: str = ""
    cot_buf: str = ""


def _greedy_token(logits_row: torch.Tensor) -> int:
    return int(torch.argmax(logits_row, dim=-1).item())


def grade_prompts(
    gen,
    enc,
    prompts: List[List[int]],
    *,
    max_new: int = 2048,
    stop_tokens: Optional[set[int]] = None,
    on_finish: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Run a batch of prompts through BatchedGenerator and parse strict JSON from final channel.

    Args:
        gen: nmoe.data.model.BatchedGenerator (constructed on target checkpoint)
        enc: Harmony encoding (already loaded)
        prompts: list[list[int]] Harmony-formatted token IDs
        max_new: decode budget per request
        stop_tokens: optional override; defaults to enc.stop_tokens_for_assistant_actions()

    Returns per-prompt dicts (same order):
        {"ok": bool, "scores": dict|None, "final_text": str, "cot_len": int}
    """
    stop: set[int] = stop_tokens or set(enc.stop_tokens_for_assistant_actions())

    # Submit all requests and track original order
    active: Dict[int, _SeqState] = {}
    order: Dict[int, int] = {}
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

    for i, toks in enumerate(prompts):
        sid = gen.add(toks, max_tokens=max_new)
        active[sid] = _SeqState(parser=StreamableParser(enc, role=Role.ASSISTANT))
        order[sid] = i

    # Decode loop
    while True:
        if not active:
            break
        out = gen.step()
        if out is None:
            # If generator is idle, drain leftovers and break
            if getattr(gen, "idle", True):
                break
            else:
                continue

        logits, sids = out
        for i, sid in enumerate(sids):
            st = active.get(sid)
            if st is None:
                continue
            tok = _greedy_token(logits[i])
            prev_channel = st.parser.current_channel
            parser_ok = True
            try:
                st.parser.process(tok)
                if st.parser.last_content_delta:
                    if st.parser.current_channel == "final":
                        st.final_buf += st.parser.last_content_delta
                    elif st.parser.current_recipient is None:
                        st.cot_buf += st.parser.last_content_delta
            except Exception:
                parser_ok = False

            is_stop = tok in stop
            is_final_stop = (prev_channel == "final") and is_stop and parser_ok
            gen.update(sid, tok, finished=is_final_stop)

            if is_final_stop:
                text = (st.final_buf or "").strip()
                rec: Dict[str, Any]
                try:
                    parsed = json.loads(text)
                    rec = {"ok": True, "scores": parsed, "final_text": text, "cot_len": len(st.cot_buf)}
                except Exception:
                    rec = {"ok": False, "scores": None, "final_text": text, "cot_len": len(st.cot_buf)}
                idx = order[sid]
                results[idx] = rec
                if on_finish is not None:
                    try:
                        on_finish(idx, rec)
                    except Exception:
                        pass
                del active[sid]

    # EOS salvage for any lingering sequences (e.g., hit max_new without a stop)
    if active:
        for sid, st in list(active.items()):
            try:
                st.parser.process_eos()
            except Exception:
                pass
            text = (st.final_buf or "").strip()
            rec: Dict[str, Any]
            try:
                parsed = json.loads(text)
                rec = {"ok": True, "scores": parsed, "final_text": text, "cot_len": len(st.cot_buf)}
            except Exception:
                rec = {"ok": False, "scores": None, "final_text": text, "cot_len": len(st.cot_buf)}
            idx = order[sid]
            results[idx] = rec
            if on_finish is not None:
                try:
                    on_finish(idx, rec)
                except Exception:
                    pass
            del active[sid]

    # Replace any None (should not happen) with fail records
    for i, r in enumerate(results):
        if r is None:
            results[i] = {"ok": False, "scores": None, "final_text": "", "cot_len": 0}

    return results  # type: ignore[return-value]


# ============================================================================
# Aggregation helpers
# ============================================================================

def compute_aggregated(scores: Dict[str, float]) -> float:
    """Compute aggregated scalar on [0,1] from general 0–4 dimensions.

    Uses keys present in the intersection with the expected set to be robust.
    """
    if not scores:
        return 0.0
    dims = ["helpfulness", "correctness", "coherence", "complexity", "density"]
    vals: List[float] = []
    for k in dims:
        if k in scores:
            vals.append(max(0.0, min(4.0, float(scores[k]))) / 4.0)
    return (sum(vals) / len(vals)) if vals else 0.0


def compute_aggregated_code(scores: Dict[str, float]) -> float:
    """Compute aggregated scalar on [0,1] for code rubric (0–10 dims)."""
    if not scores:
        return 0.0
    dims = ["readability", "modularity", "clarity", "reusability"]
    vals: List[float] = []
    for k in dims:
        if k in scores:
            vals.append(max(0.0, min(10.0, float(scores[k]))) / 10.0)
    return (sum(vals) / len(vals)) if vals else 0.0
