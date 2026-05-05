"""Math answer extraction + grading + 4-regime reward for un-grpo-maj.

Parallels `projects/co-grpo-dp/co_label_utils.py`. Extraction and grading are
powered by the `verifiers/qwen/` package (Apache-2.0). Functions here are
**deliberately byte-identical to the co-grpo-dp copy** per the repo's
"trainer self-contained, share by copy" convention. When updating one, mirror
the change in the other.
"""

from collections import Counter

from verifiers.qwen.qwen_math_parser import extract_answer
from verifiers.qwen.math_normalize import normalize_answer as _qwen_normalize
from verifiers.qwen.math_grade import grade_answer


# Sentinel written into `solution` for prompt groups that fail the self-
# consistency threshold. Cannot match any parsed answer (qwen.grade_answer
# returns False on this string), so reward evaluates to 0 for every rollout
# in such a group — no reward-function change needed.
_UNLABELED_SENTINEL = "\x00__unlabeled__\x00"


def extract_boxed_answer(text):
    """Extract the math answer from a model completion.

    Returns the answer string (e.g. ``\\frac{1}{2}``, ``42``) or ``None`` if no
    parseable answer is found. Backed by Qwen2.5-Math's parser, which handles
    \\boxed{}, "answer is X", ``\\boxed`` without braces, and many fallbacks.
    """
    return extract_answer(text, "math")


def normalize_answer(answer):
    """Canonicalize an answer for use as a hash key in majority-vote grouping."""
    return _qwen_normalize(answer)


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _extract_and_normalize(completion):
    """Extract from text, then canonicalize. Returns ``None`` on failure or empty."""
    result = normalize_answer(extract_boxed_answer(_get_text(completion)))
    if result is None or result == "":
        return None
    return result


def _majority_vote(answers, threshold):
    """Group N rollouts of one prompt by canonical answer; return the plurality."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0
    counts = Counter(valid)
    top_answer, top_count = counts.most_common(1)[0]
    top_freq = top_count / len(valid)
    if top_freq < threshold:
        return None, top_freq
    return top_answer, top_freq


def compute_4regime_reward(my_answer, peer_answers, tau_high, tau_mid, lambda_):
    """Confidence-gated 4-regime reward for one rollout.

    Builds an empirical distribution from `peer_answers` (treating ``None`` as
    unparseable) and scores `my_answer` against four regimes determined by the
    peer top-1 confidence and the rollout-answer frequency in the peer.

    Frequencies are computed over the full `peer_answers` length ``N`` (not over
    the valid-only count). This means a peer batch where only 2 of 8 rollouts
    parsed will yield ``c_top = 2/8 = 0.25`` even if both name the same answer —
    intentionally, since 2 valid samples are not enough evidence for high
    confidence. (`_majority_vote` above uses ``top_count / valid_count`` for a
    different purpose: deciding whether the plurality clears a self-consistency
    threshold for pseudo-label acceptance.)

    Args:
        my_answer (`str` or `None`):
            The canonicalized answer from this rollout. ``None`` means
            extraction failed for this rollout — treated as ``unseen``.
        peer_answers (`list[str | None]`):
            Length-``N`` list of canonicalized answers from the peer (or self,
            for un-grpo-maj). ``None`` entries are unparseable peer rollouts.
        tau_high (`float`):
            Confident-positive threshold. ``c_top >= tau_high`` enables ``+1``.
        tau_mid (`float`):
            Strong-runner-up threshold. ``p_my >= tau_mid`` (and ``my_answer !=
            top``) maps to reward ``0``.
        lambda_ (`float`):
            Negative penalty magnitude for weak / unseen rollouts. The returned
            reward in those regimes is ``-lambda_``.

    Returns:
        `tuple[float, str]`: ``(reward, regime)``.

        ``regime`` is one of:
          - ``"confident_positive"``  : ``my == top`` and ``c_top >= tau_high`` (reward = ``+1``)
          - ``"uncertain_majority"``  : ``my == top`` and ``c_top  < tau_high`` (reward = ``0``)
          - ``"strong_runner_up"``    : ``my != top`` and ``p_my >= tau_mid``   (reward = ``0``)
          - ``"weak"``                : ``my != top`` and ``0 < p_my < tau_mid``(reward = ``-lambda_``)
          - ``"unseen"``              : ``my != top`` and ``p_my == 0``         (reward = ``-lambda_``)
                                        (also when ``my_answer is None``)
          - ``"no_valid_peer"``       : ``peer_answers`` empty or all ``None``  (reward = ``0``)
          - ``"tied_top"``            : multiple peer answers share the maximum count (reward = ``0``)

    Tied-top short-circuit: when 2+ answers share the max count in `peer_answers`,
    the "top" assignment via `Counter.most_common` would be order-dependent (first-
    inserted wins), producing arbitrary asymmetric rewards across tied-top rollouts.
    All such groups return ``(0.0, "tied_top")`` instead — the peer signal is treated
    as unusable, equivalent in spirit to ``no_valid_peer``. Fires regardless of
    ``my_answer`` (so it preempts the ``my_answer is None`` -> unseen branch).
    """
    if not peer_answers:
        return 0.0, "no_valid_peer"
    N = len(peer_answers)
    valid_peer = [a for a in peer_answers if a is not None]
    if not valid_peer:
        return 0.0, "no_valid_peer"

    counts = Counter(valid_peer)
    top_answer, top_count = counts.most_common(1)[0]

    # Tied top -> peer signal is order-dependent noise; short-circuit to no signal.
    if sum(1 for c in counts.values() if c == top_count) > 1:
        return 0.0, "tied_top"

    c_top = top_count / N

    if my_answer is None:
        return -lambda_, "unseen"

    p_my = counts.get(my_answer, 0) / N

    if my_answer == top_answer:
        if c_top >= tau_high:
            return 1.0, "confident_positive"
        return 0.0, "uncertain_majority"
    if p_my == 0:
        return -lambda_, "unseen"
    if p_my >= tau_mid:
        return 0.0, "strong_runner_up"
    return -lambda_, "weak"


__all__ = [
    "extract_boxed_answer",
    "normalize_answer",
    "_extract_and_normalize",
    "_majority_vote",
    "_UNLABELED_SENTINEL",
    "compute_4regime_reward",
    "grade_answer",
    "extract_answer",
]
