"""Log-Distillation cross-supervision reward (paper-plan §3 Method 4 "分布" variant).

For each rollout y_i of policy A, the reward is the per-token average log-
probability assigned to y_i by peer policy B (length-normalized):

    r(y_i, q) = (1/T) * sum_t log p_B(y_{i,t} | q, y_{i,<t})

with epsilon-floor fallback when the peer's forward fails / returns None:

    r_fallback = log(epsilon)

## Justification (policy-gradient distillation toward peer)

Maximizing E_{y ~ π_A}[ log p_B(y | q) ] under policy gradient pulls π_A toward
high-probability regions of p_B. Not exactly KL(π_A || p_B) minimization
because π_A's entropy is not held constant, but the signal direction matches
distillation: A imitates B at the token level.

| A's rollout (per-token avg log p_B) | r(y) length-normalized |
|-------------------------------------|------------------------|
| confident peer (top1 each step)     | -0.05 to -0.5          |
| peer somewhat surprised             | -1 to -5               |
| out of distribution                 | -log(eps) ≈ -23 (cap)  |

## Trainer integration (co-grpo-dp branch)

`CoGRPOdp4RegimeTrainer._calculate_rewards_log_distill` (added 2026-05-19):
  1. Each group gathers its rollout token sequences globally (`gather_object`).
  2. Rendezvous Exchange A (`mode='train_tokens'`): swap token sequences with peer.
  3. Each group forwards peer's tokens through OWN model via
     `_get_per_token_logps_and_entropies(compute_entropy=False)` (chunked).
     Sum per-token log p over completion positions → per-rollout scalar.
  4. Rendezvous Exchange B (`mode='train_logps'`): swap sum-logp scalars with peer.
  5. Each rank slices peer's reply for its own rollouts and injects
     `inputs[i]["peer_log_prob_sum"]` + `inputs[i]["completion_lens"]`.

## Eval mode

Trainer short-circuits: `peer_log_prob_sum` is not written. The closure detects
missing kwargs and falls back to binary equality against `solution`, matching
baseline `reward_correctness`.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co_label_utils import extract_boxed_answer, grade_answer


_DEFAULT_EPSILON = 1e-10


def _get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _eval_binary_against_gt(completions, solution):
    """Eval-mode fallback: 1.0 if extract_boxed(completion) sympy-equals solution, else 0.0."""
    rewards = []
    for completion, gt in zip(completions, solution):
        text = _get_text(completion)
        pred = extract_boxed_answer(text) if text else None
        rewards.append(1.0 if (pred is not None and grade_answer(pred, gt)) else 0.0)
    return rewards


def compute_log_distill_reward(
    peer_log_prob_sums,
    completion_lens,
    epsilon: float = _DEFAULT_EPSILON,
):
    """Length-normalized log-distill reward.

    Args:
        peer_log_prob_sums: list[float | None] — per-rollout total log p_PEER(my_y)
            over completion tokens. `None` indicates peer's forward failed
            (tokenization mismatch / crash) and trainer fell back. Reward then
            uses log(epsilon).
        completion_lens: list[int] — per-rollout completion length.
        epsilon: token-probability floor for the None / zero-length fallback.

    Returns:
        list[float]: per-rollout reward = peer_log_prob_sums[i] / completion_lens[i]
        (per-token average log p_PEER), or log(epsilon) on fallback.
    """
    fallback = math.log(epsilon)
    rewards = []
    for logp_sum, T in zip(peer_log_prob_sums, completion_lens):
        if logp_sum is None or T == 0:
            rewards.append(fallback)
        else:
            rewards.append(float(logp_sum) / int(T))
    return rewards


def make_reward_log_distill(epsilon: float = _DEFAULT_EPSILON):
    """Closure for `--reward_type log_distill`.

    Trainer injects `peer_log_prob_sum` (list[float|None]) and `completion_lens`
    (list[int]) into `inputs[i]`. This closure surfaces them to TRL after
    aggregation. In eval mode the closure detects missing kwargs and falls back
    to binary equality against `solution`.
    """

    def reward_log_distill(
        completions,
        peer_log_prob_sum=None,
        completion_lens=None,
        solution=None,
        log_metric=None,
        **kwargs,
    ):
        if peer_log_prob_sum is None or completion_lens is None:
            return _eval_binary_against_gt(completions, solution)
        rewards = compute_log_distill_reward(peer_log_prob_sum, completion_lens, epsilon)
        if log_metric is not None and rewards:
            log_metric("log_distill/avg_reward", sum(rewards) / len(rewards))
            log_metric("log_distill/min_reward", min(rewards))
            log_metric("log_distill/max_reward", max(rewards))
            n_fallback = sum(1 for lp in peer_log_prob_sum if lp is None)
            log_metric("log_distill/fraction_fallback", n_fallback / len(rewards))
        return rewards

    return reward_log_distill


__all__ = [
    "compute_log_distill_reward",
    "make_reward_log_distill",
]
