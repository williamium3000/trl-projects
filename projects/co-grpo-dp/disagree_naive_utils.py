"""Method 2 (disagreement-weighted) and Method 3 (naive soft proportion) reward
factories for co-grpo-dp.

Both rewards consume the same trainer-side payload that
`co_grpo_dp_4regime_trainer.CoGRPOdp4RegimeTrainer` already exposes via
`inputs[i]["peer_answers"]` — a length-`num_generations` list of canonicalized
peer answers for the prompt this rollout belongs to. No trainer change is
needed; only the reward function differs.

Both factories return a closure that dispatches on `peer_answers`:
  - train mode (peer_answers is a list of length len(completions)): score per
    Method 2 / Method 3 against the peer's per-prompt answer list.
  - eval mode (peer_answers is None — trainer didn't write it): fall back to
    binary 1/0 against the dataset's ground-truth `solution`. Identical eval
    semantics to baseline `reward_correctness` and `reward_4regime`.

Method 2 (`make_reward_disagree`) wraps a `base_reward_fn` and multiplies it by
a per-prompt scalar w(q) ∈ [0,1] that is small when the two groups' answer
distributions agree (cross-supervision degenerates to self-supervision) and
large when they disagree. Three variants of w(q) are supported:
  - top1: 1 if argmax differs, else 1 - c_A * c_B
  - tv:   total variation between the two empirical answer distributions
  - jsd:  Jensen-Shannon divergence (normalized by log 2 → range [0,1])

Method 3 (`make_reward_naive`) returns r(y) = p_A(y), i.e. the relative
frequency with which `my_answer` appears in the peer's rollouts. Reward is 0
when `my_answer` is None (extraction failed) or unseen in the peer.
"""

import math
from collections import Counter

from co_label_utils import _extract_and_normalize, extract_boxed_answer, grade_answer


def _get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _empirical_distribution(answers):
    """Return (Counter, N) where N is the full length and Counter excludes None.

    Probabilities are computed against `N` (not the valid count) so that prompts
    with many unparseable rollouts produce damped distributions — matching the
    convention used by `compute_4regime_reward` and `_majority_vote` for the
    confidence-frequency case.
    """
    N = len(answers)
    counts = Counter(a for a in answers if a is not None)
    return counts, N


def compute_disagreement_weight(my_counts, my_N, peer_counts, peer_N, variant, w_min=0.0):
    """Compute the per-prompt disagreement weight w(q) ∈ [0, 1].

    Args:
        my_counts (`Counter`):
            Counter of canonicalized answers from this group's rollouts (excl. None).
        my_N (`int`):
            Total rollouts in this group's prompt batch (incl. unparseable).
        peer_counts (`Counter`):
            Counter of canonicalized peer answers (excl. None).
        peer_N (`int`):
            Total rollouts in the peer's prompt batch (incl. unparseable).
        variant (`str`):
            One of `"top1"`, `"tv"`, `"jsd"`. See module docstring for definitions.
        w_min (`float`):
            Floor applied as `w = max(w, w_min)`. Prevents weights from collapsing
            to 0 on transient low-disagreement prompts. Default 0.0 disables.

    Returns:
        `float`: w(q) ∈ [w_min, 1].

    Edge cases:
      - Either side has no parseable rollouts (both Counters empty): returns 1.0
        (no signal to penalize; let base_reward decide).
      - One side empty, other not: returns 1.0 (asymmetric → maximally informative).
    """
    if my_N == 0 or peer_N == 0:
        return max(1.0, w_min)
    if not my_counts and not peer_counts:
        return max(1.0, w_min)
    if not my_counts or not peer_counts:
        return max(1.0, w_min)

    if variant == "top1":
        my_top, my_top_count = my_counts.most_common(1)[0]
        peer_top, peer_top_count = peer_counts.most_common(1)[0]
        if my_top != peer_top:
            w = 1.0
        else:
            c_my = my_top_count / my_N
            c_peer = peer_top_count / peer_N
            w = 1.0 - c_my * c_peer
    elif variant == "tv":
        support = set(my_counts) | set(peer_counts)
        w = 0.5 * sum(abs(my_counts.get(y, 0) / my_N - peer_counts.get(y, 0) / peer_N) for y in support)
    elif variant == "jsd":
        support = set(my_counts) | set(peer_counts)
        jsd = 0.0
        for y in support:
            p = my_counts.get(y, 0) / my_N
            q = peer_counts.get(y, 0) / peer_N
            m = 0.5 * (p + q)
            if m == 0.0:
                continue
            if p > 0:
                jsd += 0.5 * p * math.log(p / m)
            if q > 0:
                jsd += 0.5 * q * math.log(q / m)
        w = jsd / math.log(2.0)
    else:
        raise ValueError(f"Unknown disagreement variant: {variant!r}")

    return max(min(w, 1.0), w_min)


def compute_naive_soft_reward(my_answer, peer_answers):
    """Method 3: r(y) = p_A(y), the empirical frequency of `my_answer` in `peer_answers`.

    Args:
        my_answer (`str` or `None`):
            Canonicalized answer from this rollout. None → reward 0 (no signal).
        peer_answers (`list[str | None]`):
            Length-N list of canonicalized peer answers; None entries mean
            unparseable peer rollouts and depress the denominator.

    Returns:
        `tuple[float, str]`: ``(reward, regime)`` where regime is one of
        ``"naive_hit"`` (peer count > 0), ``"naive_unseen"`` (in peer but count = 0),
        ``"naive_my_none"`` (my_answer is None), ``"no_valid_peer"``.
    """
    if not peer_answers:
        return 0.0, "no_valid_peer"
    N = len(peer_answers)
    if my_answer is None:
        return 0.0, "naive_my_none"
    counts = Counter(a for a in peer_answers if a is not None)
    if not counts:
        return 0.0, "no_valid_peer"
    p = counts.get(my_answer, 0) / N
    if p == 0:
        return 0.0, "naive_unseen"
    return p, "naive_hit"


def make_reward_disagree(variant, w_min, base_reward_fn):
    """Method 2: r_final(y, q) = w(q) * base_reward(y, q).

    Args:
        variant (`str`):
            One of `"top1"`, `"tv"`, `"jsd"`.
        w_min (`float`):
            Lower floor for w(q) to prevent total signal collapse on borderline
            prompts. 0.0 disables.
        base_reward_fn (`callable`):
            A reward closure with the same signature as `reward_correctness` /
            `reward_4regime`: `(completions, peer_answers=None, solution=None,
            log_metric=None, **kwargs) -> list[float]`. The base reward is what
            gets multiplicatively scaled. In eval mode the wrapper short-circuits
            to base_reward_fn (peer_answers is None there) so eval accuracy is
            unaffected.

    Returns:
        `callable`: a reward closure with the standard signature.
    """

    def reward_disagree(completions, peer_answers=None, solution=None, log_metric=None, **kwargs):
        if peer_answers is None:
            return base_reward_fn(completions, peer_answers=None, solution=solution, log_metric=log_metric, **kwargs)

        base_rewards = base_reward_fn(
            completions, peer_answers=peer_answers, solution=solution, log_metric=log_metric, **kwargs
        )

        # Group rollouts by their peer_answers identity. Within one prompt the
        # peer_answers list is shared across all G rollouts, so each unique
        # peer_answers tuple corresponds to one prompt (modulo the rare case of
        # two prompts producing identical peer answer multisets — harmless: they
        # get the same w(q) which is what we want).
        rewards = []
        weights_logged = []
        # Cache w(q) per unique peer answer tuple to avoid recomputation across G rollouts.
        weight_cache = {}
        # We also need the *self* (my-side) distribution. The reward function is
        # called per-rollout but receives the full `completions` batch, so we
        # build my distributions per-prompt by collecting the rollouts that share
        # a peer_answers list.
        prompt_to_my_answers = {}
        prompt_to_peer = {}
        for i, peer in enumerate(peer_answers):
            key = tuple(peer) if peer is not None else None
            prompt_to_my_answers.setdefault(key, []).append(_extract_and_normalize(completions[i]))
            prompt_to_peer[key] = peer

        for i, peer in enumerate(peer_answers):
            key = tuple(peer) if peer is not None else None
            if key in weight_cache:
                w = weight_cache[key]
            else:
                my_answers = prompt_to_my_answers[key]
                my_counts, my_N = _empirical_distribution(my_answers)
                peer_counts, peer_N = _empirical_distribution(prompt_to_peer[key] or [])
                w = compute_disagreement_weight(my_counts, my_N, peer_counts, peer_N, variant, w_min)
                weight_cache[key] = w
            rewards.append(w * base_rewards[i])
            weights_logged.append(w)

        if log_metric is not None and weights_logged:
            log_metric(f"disagree/avg_weight_{variant}", sum(weights_logged) / len(weights_logged))
            log_metric(f"disagree/min_weight_{variant}", min(weights_logged))
            log_metric(f"disagree/max_weight_{variant}", max(weights_logged))

        return rewards

    return reward_disagree


def make_reward_naive():
    """Method 3: r(y) = p_A(y).

    Returns:
        `callable`: a reward closure with signature
        `(completions, peer_answers=None, solution=None, log_metric=None, **kwargs) -> list[float]`.
    """

    def reward_naive(completions, peer_answers=None, solution=None, log_metric=None, **kwargs):
        rewards = []

        if peer_answers is None:
            for completion, gt in zip(completions, solution):
                pred_raw = extract_boxed_answer(_get_text(completion))
                rewards.append(1.0 if pred_raw is not None and grade_answer(pred_raw, gt) else 0.0)
            return rewards

        regime_counts = {"naive_hit": 0, "naive_unseen": 0, "naive_my_none": 0, "no_valid_peer": 0}
        for completion, peer in zip(completions, peer_answers):
            pred_canonical = _extract_and_normalize(completion)
            r, regime = compute_naive_soft_reward(pred_canonical, peer)
            rewards.append(r)
            regime_counts[regime] += 1

        if log_metric is not None and len(completions) > 0:
            total = len(completions)
            for regime, cnt in regime_counts.items():
                log_metric(f"naive/pct_{regime}", cnt / total)
            log_metric("naive/avg_reward", sum(rewards) / total)

        return rewards

    return reward_naive


def make_reward_binary():
    """Binary correctness against peer majority — same shape as Method 2's base reward option.

    For Method 2 with `--disagree_base_reward binary`: the inner reward returns
    1.0 if `my_answer` matches the peer's plurality (treating empty / all-None
    peer as 0), else 0.0. Eval mode falls back to ground-truth grading.
    """

    def reward_binary(completions, peer_answers=None, solution=None, log_metric=None, **kwargs):
        rewards = []

        if peer_answers is None:
            for completion, gt in zip(completions, solution):
                pred_raw = extract_boxed_answer(_get_text(completion))
                rewards.append(1.0 if pred_raw is not None and grade_answer(pred_raw, gt) else 0.0)
            return rewards

        for completion, peer in zip(completions, peer_answers):
            pred_canonical = _extract_and_normalize(completion)
            if not peer or pred_canonical is None:
                rewards.append(0.0)
                continue
            valid_peer = [a for a in peer if a is not None]
            if not valid_peer:
                rewards.append(0.0)
                continue
            counts = Counter(valid_peer)
            top_answer, top_count = counts.most_common(1)[0]
            # Tied top: peer signal is order-dependent noise, treat as no signal.
            if sum(1 for c in counts.values() if c == top_count) > 1:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if pred_canonical == top_answer else 0.0)

        return rewards

    return reward_binary


__all__ = [
    "compute_disagreement_weight",
    "compute_naive_soft_reward",
    "make_reward_disagree",
    "make_reward_naive",
    "make_reward_binary",
]
