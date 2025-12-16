"""
K2-style knowledge rephrasing.

Usage:
    prompt = format_prompt(text)
    parser = create_parser()
    stops = stop_tokens()
    final = ""

    for token in generate(prompt):
        parser.process(token)
        if parser.current_channel == "final" and parser.last_content_delta:
            final += parser.last_content_delta
"""
import torch
from openai_harmony import (
    Conversation, HarmonyEncodingName, Message, Role,
    StreamableParser, load_harmony_encoding,
)

_encoding = None
DEFAULT_INSTRUCTION = "Rewrite this text in different words while preserving all information."


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _encoding


def format_prompt(text: str, instruction: str = DEFAULT_INSTRUCTION) -> list[int]:
    """Format text as Harmony chat prompt for rephrasing."""
    enc = _get_encoding()
    conv = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, instruction),
        Message.from_role_and_content(Role.USER, text),
    ])
    return enc.render_conversation_for_completion(conv, Role.ASSISTANT)


def create_parser() -> StreamableParser:
    """Create a StreamableParser for streaming token processing."""
    return StreamableParser(_get_encoding(), role=Role.ASSISTANT)


def stop_tokens() -> set[int]:
    """Get Harmony stop tokens for generation."""
    return set(_get_encoding().stop_tokens_for_assistant_actions())


def sample_top_p(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.9) -> int:
    """Sample from logits using top-p (nucleus) sampling."""
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_indices[idx].item()
