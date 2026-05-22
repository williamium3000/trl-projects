"""Custom-task helpers for AIME-2025 / AMC23 lm-eval-harness yamls.

We extract the **last** \\boxed{...} from the model completion and compare to
the gold answer. Comparison strategy:

  - AIME 2025: integer-only answers in 0..999. Try int parse on both sides.
  - AMC 23: numeric / symbolic (e.g. "1/2", "\\sqrt{3}"). Use `math_verify`
            (sympy-based, latex-aware) for equivalence.

Both `process_results_*` functions return:
    {"exact_match": 1.0 | 0.0}

so they slot into the yaml's `metric_list: [{metric: exact_match, ...}]`.
"""

from __future__ import annotations

import re
from typing import Any


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+|\{[^}]*\})\}")


def _last_boxed(text: str) -> str | None:
    """Return content of the *last* \\boxed{...} in `text`, or None.

    Handles 1-level nested braces (e.g. \\boxed{\\frac{1}{2}}).
    """
    if not text:
        return None
    # Greedy walk to find the last \boxed{ and balance braces.
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # fallback: simple regex (no nesting)
        m = _BOXED_RE.findall(text)
        return m[-1].strip() if m else None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if out else None


def process_results_aime(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """AIME-2025: integer answer 000..999."""
    completion = results[0] if results else ""
    pred = _last_boxed(completion)
    target = str(doc.get("answer", "")).strip()

    if pred is None:
        return {"exact_match": 0.0}

    # Strip $ ... $ wrappers, leading 0s, trailing punctuation.
    pred_clean = pred.strip().strip("$").strip()

    try:
        return {"exact_match": float(int(pred_clean) == int(target))}
    except (ValueError, TypeError):
        return {"exact_match": float(pred_clean == target)}


def process_results_amc(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """AMC-23: numeric / symbolic; use math_verify for equivalence."""
    completion = results[0] if results else ""
    pred = _last_boxed(completion)
    target = str(doc.get("answer", "")).strip()

    if pred is None:
        return {"exact_match": 0.0}

    try:
        from math_verify import parse, verify  # type: ignore
        gold = parse(f"${target}$")
        guess = parse(f"${pred}$")
        return {"exact_match": float(bool(verify(gold, guess)))}
    except Exception:
        # Fallback: best-effort numeric compare.
        try:
            return {"exact_match": float(float(pred) == float(target))}
        except (ValueError, TypeError):
            return {"exact_match": float(pred.strip() == target.strip())}
