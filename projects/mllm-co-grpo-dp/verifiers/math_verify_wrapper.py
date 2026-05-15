"""math_verify-based grading for mllm-co-grpo-dp.

Replaces the qwen-sympy verifier used in co-grpo-dp / co-grpo / un-grpo-maj
/ grpo. Uses HuggingFace's `math_verify` (the same grader R1-V baseline
uses), which handles:
  - LaTeX form equivalence (`\\frac{1}{2}` == `0.5` == `1/2`)
  - Unit stripping (`140°` parses as `140`, `2.4cm` as `2.4`)
  - Symbolic forms (`\\pi`, `\\sqrt{2}`)
  - Degree/radian conversion

Three callers share this module, mirroring co-grpo-dp's three-way design:

1. `_majority_vote` in `co_label_utils.py` (uses `normalize_answer` as a
   fast hash key for clustering, no sympy cost).
2. `reward_correctness` in `train_mllm_co_grpo_dp.py` (uses `grade_answer`
   for true math equivalence — robust to format diffs in peer pseudo-labels).
3. `mllm_co_grpo_dp_trainer._calculate_rewards` eval-mode short-circuit
   (delegates to parent reward path, so eval accuracy uses the same
   `grade_answer` against the dataset's ground-truth solution).

⚠️ This project's `requirements.txt` pulls `latex2sympy2_extended` (antlr4
4.13.2 line), which is incompatible with the `latex2sympy2` 1.9.1 (antlr4
4.7.2) used by the marti env. Install only into the `mllm-cogrpodp` env.
See `INSTALL.md` §0 for the full env-isolation rationale.
"""

import re

from math_verify import parse, verify


# Match <answer>...</answer> tags (R1-V baseline convention). Permissive:
# - case-insensitive
# - DOTALL so the content can span newlines (thinking traces often do)
# - non-greedy `.*?` so multiple <answer> blocks each match independently;
#   we take the last one (most likely the final answer after retries).
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_tag(text):
    """Extract content from the last `<answer>...</answer>` tag in `text`.

    R1-V baseline prompts request `<think>...</think><answer>...</answer>`
    format; we extract from the answer tag (not `\\boxed{}` which is the
    math convention used by co-grpo-dp).

    Args:
        text (`str` or `None`):
            Model completion text (single turn, after _get_text unwrap).

    Returns:
        `str` or `None`: stripped answer content, or `None` if no tag is
        found, content is empty, or input is `None`.
    """
    if text is None:
        return None
    matches = _ANSWER_TAG_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    if not answer:
        return None
    return answer


def normalize_answer(answer):
    """Canonical-string normalization for majority-vote hash keys.

    Minimal: lowercase + collapse whitespace + strip. Two semantically
    equivalent answers may still hash to different buckets here (e.g.
    `1/2` vs `0.5`), but the downstream `grade_answer` uses math_verify
    semantic equivalence and recovers correctness at reward time. The
    fast path here avoids paying math_verify cost (~1-10ms per parse)
    on every cluster decision.

    Args:
        answer (`str` or `None`):
            Raw extracted answer.

    Returns:
        `str` or `None`: canonical-string form, or `None` for empty/None input.
    """
    if answer is None:
        return None
    s = " ".join(answer.lower().split())
    if not s:
        return None
    return s


def grade_answer(pred, gold):
    """Math-equivalence judgment via HuggingFace `math_verify`.

    Used by:
      - `reward_correctness` in train script (per-rollout reward)
      - eval-mode reward path (against dataset ground truth)

    Args:
        pred (`str` or `None`):
            Predicted answer extracted from a model completion.
        gold (`str` or `None`):
            Reference answer: peer pseudo-label (train) or dataset
            ground truth (eval). Can also be `_UNLABELED_SENTINEL` for
            prompt groups the peer dropped — math_verify will fail to
            parse it and return `False`, which is the desired behavior.

    Returns:
        `bool`: `True` iff `pred` and `gold` are math-equivalent under
        math_verify. Returns `False` for `None` / empty / unparseable
        inputs (after a case-insensitive exact-match fallback).
    """
    if pred is None or gold is None:
        return False
    pred = pred.strip()
    gold = gold.strip()
    if not pred or not gold:
        return False
    try:
        gold_parsed = parse(gold)
        pred_parsed = parse(pred)
    except Exception:
        # math_verify raised — fall through to string equality.
        return pred.lower() == gold.lower()
    if not gold_parsed or not pred_parsed:
        # Parsed to empty list (math_verify could not find an expr).
        # Fall back to case-insensitive exact match — catches the easy
        # cases where pred and gold are byte-identical strings but
        # not valid math expressions (e.g., GEOQA ambiguous shapes).
        return pred.lower() == gold.lower()
    try:
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return pred.lower() == gold.lower()


__all__ = [
    "extract_answer_tag",
    "normalize_answer",
    "grade_answer",
]
