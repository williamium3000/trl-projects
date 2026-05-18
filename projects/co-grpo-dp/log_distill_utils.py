"""Log-Distillation cross-supervision reward (paper-plan §3 Method 2 "分布" variant).

For each rollout y_i of policy A, the reward is the sequence log-probability
assigned to y_i under peer policy B:

    r(y_i, q) = log p_B(y_i | q)

with epsilon smoothing at the token level to prevent -inf when p_B assigns
near-zero probability to an A-generated token:

    log p_B(y_i | q) = sum_t log( max(p_B(y_{i,t} | q, y_{i,<t}), epsilon) )

## Theoretical justification

Maximizing E_{y ~ pi_A}[ log p_B(y | q) ] is exactly minimizing
KL(pi_A || p_B) (up to constant in pi_A's entropy). This aligns A's output
distribution to B's empirical distribution — a principled upgrade over the
naive "soft proportion" reward (Method 3) which just uses peer frequency of
A's answer string.

| A 答 | Soft proportion | Log distillation |
|------|-----------------|------------------|
| top1 (8/16) | 0.500           | -0.69            |
| top2 (7/16) | 0.438           | -0.83            |
| fluke (1/16)| 0.063           | -2.77            |
| unseen      | 0.000           | -log(1/eps) ~ -23 (eps=1e-10) |

Low-frequency / fluke answers get exponentially-punished; soft proportion only
penalizes them linearly. → log_distill naturally suppresses B's long-tail noise.

## Dependency on co-grpo-dp infra (scaffold gap)

This reward requires peer-model B to forward-pass A's rollout token sequences
and return sequence log-probabilities. The existing rendezvous payload is just
answer strings (`list[str | None]` of length G per prompt). To enable
log_distillation, the rendezvous protocol needs an additional channel:

  Group A → B:  G token-id sequences per prompt + attention masks
  Group B → A:  G scalar log p_B(y_i) values per prompt

Implementation TODO:
  1. Add `--peer_logprob_payload` mode to `Rendezvous` (rendezvous.py).
  2. In `_calculate_rewards` of `co_grpo_dp_4regime_trainer.py`, after my
     group's rollouts are tokenized but BEFORE the group A→B exchange:
       - Group B receives A's token sequences from rendezvous.
       - Group B does a forward pass through self.model (no_grad) on (q, y_i)
         using `_get_per_token_logps_and_entropies` (already exists in
         trainer base for KL computation).
       - Group B sums per-token log p across completion mask, applies eps,
         sends back the per-prompt scalar list.
       - Group A receives back log p_B(y_i) per rollout, injects into
         inputs[i]["peer_log_prob"].
  3. Reward function below consumes `peer_log_prob` from kwargs.

Estimated effort: ~150 lines (rendezvous extension + trainer hook + tests).
Once integrated, this is a pure plug-and-play reward via `--reward_type log_distill`.
"""

from __future__ import annotations

import math


_DEFAULT_EPSILON = 1e-10


def compute_log_distill_reward(
    peer_token_logps: list[float | None],
    completion_lens: list[int],
    epsilon: float = _DEFAULT_EPSILON,
) -> list[float]:
    """Sequence-level log p_B(y_i) reward, length-normalized.

    Args:
        peer_token_logps (`list[float | None]`):
            Per-rollout sum of per-token log p_B values. `None` indicates the
            peer group failed to compute this rollout's sequence logp (e.g.,
            tokenization mismatch or peer crash) → reward falls back to
            `log(epsilon)` to discourage A from exploring tokens B can't
            assign mass to.
        completion_lens (`list[int]`):
            Per-rollout completion length (number of tokens). Used to
            length-normalize so that long-but-low-prob and short-but-low-prob
            sequences get comparable rewards.
        epsilon (`float`, *optional*, defaults to `1e-10`):
            Token-level probability floor inside the trainer's forward pass.
            This argument controls the FALLBACK reward when peer logp is
            None — it should match the trainer-side smoothing constant.

    Returns:
        `list[float]`: per-rollout reward = `peer_token_logps[i] / completion_lens[i]`
        (per-token average log p_B), or `log(epsilon)` when peer logp is None.
    """
    fallback = math.log(epsilon)
    rewards = []
    for logp, T in zip(peer_token_logps, completion_lens):
        if logp is None or T == 0:
            rewards.append(fallback)
        else:
            rewards.append(logp / T)
    return rewards


def make_reward_log_distill(epsilon: float = _DEFAULT_EPSILON):
    """Return a closure-bound reward fn for `--reward_type log_distill`.

    Requires `peer_token_logps` (list[float|None], length len(completions))
    and `completion_lens` (list[int]) in **kwargs. Trainer must inject both.

    See module docstring for the trainer-integration plan; this function
    raises NotImplementedError when invoked without the kwargs to make the
    integration gap loud.
    """

    def reward_log_distill(completions, peer_token_logps=None, completion_lens=None,
                           solution=None, log_metric=None, **kwargs):
        if peer_token_logps is None or completion_lens is None:
            raise NotImplementedError(
                "log_distill reward requires peer_token_logps + completion_lens. "
                "Wire the co-grpo-dp rendezvous to exchange per-rollout peer "
                "sequence log-prob. See projects/co-grpo-dp/log_distill_utils.py."
            )
        rewards = compute_log_distill_reward(peer_token_logps, completion_lens, epsilon)
        if log_metric is not None and rewards:
            log_metric("log_distill/avg_reward", sum(rewards) / len(rewards))
            log_metric("log_distill/min_reward", min(rewards))
            log_metric("log_distill/max_reward", max(rewards))
        return rewards

    return reward_log_distill
