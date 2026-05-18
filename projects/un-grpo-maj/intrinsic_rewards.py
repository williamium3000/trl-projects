"""Intrinsic single-model rewards: Entropy minimization & Self-Certainty maximization.

These are baselines from Co-rewarding (arXiv 2508.00410, ICLR 2026) Table 1.
Originally from:
  - Self-Certainty: Zhao et al. 2025, "Learning to reason without external rewards"
    (arXiv 2505.19590).
  - Entropy:        Prabhudesai et al. 2025, "Maximizing confidence alone improves
    reasoning" (arXiv 2505.22660).

Both rewards operate purely on the model's own per-token softmax distribution
over the full vocabulary — they do NOT require ground-truth labels, peer
pseudo-labels, or any external signal. This makes them the natural single-view
baselines to compare against our cross-supervised (peer-view) methods.

## Math

Let p_t ∈ R^V be the model's softmax distribution over the vocab at position t,
y a generated sequence of length T, and U = 1/V the uniform distribution.

  Entropy reward (we maximize NEGATIVE entropy = we minimize entropy):
      r_entropy(y) = -1/T · sum_t H(p_t) = 1/T · sum_t sum_v p_t(v) log p_t(v)

  Self-Certainty reward (maximize KL from uniform):
      r_sc(y) = 1/T · sum_t KL(p_t || U)
              = 1/T · sum_t [log V - H(p_t)]
              = log V + r_entropy(y)

  → r_sc = log V + r_entropy, so the two rewards are affine transforms of each
  other in expectation. Their GRPO-normalized advantages will differ slightly
  because of per-batch variance, but they encode the same supervision.

## Implementation note (scaffold gap)

Both rewards require the FULL softmax over vocabulary at each position. TRL's
GRPO reward function signature is `reward_fn(completions, ...) -> list[float]`
and only receives extracted text — it does NOT pass logits. To wire these as
GRPO rewards we need EITHER:

  (a) An extra forward pass on the generated tokens inside `_calculate_rewards`
      to recover logits (memory cost: B*T*V floats per batch, ~6GB at 3B/V=152k);
  (b) Capture vLLM's `prompt_logprobs=True` or `logprobs=K` (top-K) and
      use the top-K approximation (faster, ~5% bias for K≥20).

`compute_*` below take per-token logits as input and assume the caller has
acquired them somehow. The trainer wiring is a separate TODO.

See: `projects/un-grpo-maj/self_label_4regime_trainer.py` for where to plug in.
"""

from __future__ import annotations

import math

import torch


# Numerical guards. Match TRL's clamp conventions in other trainers.
_EPS = 1e-10


def compute_entropy_reward(
    logits: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative-entropy reward (we maximize confidence ⇔ minimize H(p_t)).

    Args:
        logits (`torch.Tensor`):
            Shape `(B, T, V)`. Raw logits at each generated position. NOT
            softmax — function takes log-softmax internally for stability.
        completion_mask (`torch.Tensor`):
            Shape `(B, T)`. 1 for valid completion tokens, 0 for padding.
            Used to mask out padded positions before averaging.

    Returns:
        `torch.Tensor`: Shape `(B,)`. Per-sequence reward = mean over valid
        positions of `sum_v p_t(v) log p_t(v)` (= -H(p_t)). Larger = more
        confident (peakier distribution).
    """
    log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)
    probs = log_probs.exp()
    # Per-position negative entropy: sum_v p log p (negative, peakier=closer to 0)
    neg_entropy_per_pos = (probs * log_probs).sum(dim=-1)  # (B, T)
    # Mean over valid positions per sequence
    valid_lens = completion_mask.sum(dim=-1).clamp(min=1)
    per_seq = (neg_entropy_per_pos * completion_mask).sum(dim=-1) / valid_lens
    return per_seq


def compute_self_certainty_reward(
    logits: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Self-Certainty reward = KL(p_t || Uniform) averaged over valid positions.

    `KL(p || U) = log V + sum_v p log p = log V - H(p)`. Equivalent up to an
    additive constant to `compute_entropy_reward`, but kept as a separate
    function to match the paper's exposition and to allow downstream code to
    log them under different metric names.

    Args:
        logits (`torch.Tensor`):
            Shape `(B, T, V)`. Raw logits at each generated position.
        completion_mask (`torch.Tensor`):
            Shape `(B, T)`. 1 for valid completion tokens, 0 for padding.

    Returns:
        `torch.Tensor`: Shape `(B,)`. Per-sequence reward = mean over valid
        positions of `log V - H(p_t)`. Strictly non-negative.
    """
    vocab_size = logits.shape[-1]
    log_v = math.log(vocab_size)
    return compute_entropy_reward(logits, completion_mask) + log_v


def make_reward_entropy():
    """Return a closure-bound reward function for `--reward_type entropy`.

    The returned callable expects `per_token_logits` and `completion_mask` in
    `**kwargs` (injected by a trainer that captures them). If absent, raises
    NotImplementedError with the integration hint.
    """

    def reward_entropy(completions, per_token_logits=None, completion_mask=None,
                       solution=None, log_metric=None, **kwargs):
        if per_token_logits is None or completion_mask is None:
            raise NotImplementedError(
                "Entropy reward requires per-token logits + completion_mask. "
                "Wire your trainer to capture logits and inject them via kwargs. "
                "See projects/un-grpo-maj/intrinsic_rewards.py docstring."
            )
        rewards = compute_entropy_reward(per_token_logits, completion_mask).tolist()
        if log_metric is not None and rewards:
            log_metric("entropy/avg_reward", sum(rewards) / len(rewards))
        return rewards

    return reward_entropy


def make_reward_self_certainty():
    """Return a closure-bound reward function for `--reward_type self_certainty`.

    Same calling convention as `make_reward_entropy`.
    """

    def reward_self_certainty(completions, per_token_logits=None, completion_mask=None,
                              solution=None, log_metric=None, **kwargs):
        if per_token_logits is None or completion_mask is None:
            raise NotImplementedError(
                "Self-Certainty reward requires per-token logits + completion_mask. "
                "See projects/un-grpo-maj/intrinsic_rewards.py docstring."
            )
        rewards = compute_self_certainty_reward(per_token_logits, completion_mask).tolist()
        if log_metric is not None and rewards:
            log_metric("self_certainty/avg_reward", sum(rewards) / len(rewards))
        return rewards

    return reward_self_certainty
