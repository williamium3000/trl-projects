"""SimCT tokenizer alignment — the core of "Recovering Lost Supervision for
Cross-Tokenizer On-Policy Distillation" (arXiv 2605.07711).

Given the SAME text tokenized by two different (byte-level BPE) tokenizers, this
splits it into **minimal aligned units**: the finest spans that BOTH tokenizers
can express, i.e. the shortest sub-sequences where the two tokenizations re-sync
at a shared byte boundary.

Each emitted segment carries the token-index span on each side:
  - 1 student token ↔ 1 teacher token with the same surface string  → a *shared*
    1:1 unit (the common `V_T ∩ V_S` case);
  - otherwise (k_s student tokens ↔ k_t teacher tokens over the same bytes) → a
    *minimal aligned unit* `A`.

The two tokenizers here (Llama-3.2 / Qwen2.5) share 85% of their vocab and both
use GPT-2 byte-level encoding, so most segments are trivial 1:1 and divergences
(e.g. "42" → ['42'] vs ['4','2']) are short and local.
"""

from dataclasses import dataclass


@dataclass
class AlignedUnit:
    s_start: int   # student token index (inclusive)
    s_end: int     # student token index (exclusive)
    t_start: int
    t_end: int
    is_shared: bool  # True iff exactly one token on each side with identical string

    @property
    def s_len(self):
        return self.s_end - self.s_start

    @property
    def t_len(self):
        return self.t_end - self.t_start


def _token_byte_strings(tokenizer, token_ids):
    """Return each token's raw bytes (GPT-2 byte-level decode), so equal bytes
    across tokenizers align even when the vocab indices differ."""
    # convert_ids_to_tokens gives the byte-level surface form (e.g. 'Ġthe');
    # we compare those surface strings directly since both tokenizers use the
    # same GPT-2 byte alphabet.
    return tokenizer.convert_ids_to_tokens(list(token_ids))


def align(student_ids, teacher_ids, student_tok, teacher_tok):
    """Two-pointer alignment of the same text under two tokenizers.

    Args:
        student_ids, teacher_ids (`list[int]`): the SAME text, each tokenizer's ids.
        student_tok, teacher_tok: the two tokenizers.

    Returns:
        `list[AlignedUnit]` covering both sequences exactly, or `None` if the two
        token strings do not represent the same bytes (caller should fall back).
    """
    s_str = _token_byte_strings(student_tok, student_ids)
    t_str = _token_byte_strings(teacher_tok, teacher_ids)

    units = []
    i = j = 0
    while i < len(s_str) and j < len(t_str):
        si, tj = i, j
        s_buf = s_str[i]
        t_buf = t_str[j]
        # grow the shorter side until the accumulated byte strings match
        while s_buf != t_buf:
            if len(s_buf) < len(t_buf):
                i += 1
                if i >= len(s_str):
                    return None
                s_buf += s_str[i]
            else:
                j += 1
                if j >= len(t_str):
                    return None
                t_buf += t_str[j]
        i += 1
        j += 1
        is_shared = (i - si == 1) and (j - tj == 1)
        units.append(AlignedUnit(si, i, tj, j, is_shared))
    # both must be fully consumed for a valid alignment
    if i != len(s_str) or j != len(t_str):
        return None
    return units
