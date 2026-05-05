"""Self-supervised GRPO trainer with 4-regime confidence-gated reward.

Same mechanics as `self_label_trainer.py` (single-model, no peer, gather own
rollouts and majority-vote to derive a self-consistency signal), but instead of
overwriting `inputs[i]["solution"]` with the majority and using binary reward,
this trainer writes the **full per-prompt list of canonical answers** to
`inputs[i]["self_answers"]`. The downstream `reward_4regime` closure (defined
in `train_un_grpo_4regime.py`) consumes this distribution to score each
rollout against a confidence-gated 4-regime reward.

Eval mode short-circuits identically to baseline: `inputs[i]["self_answers"]`
is not written, `inputs[i]["solution"]` keeps the dataset value, and the
reward function falls back to binary equality against ground truth.
"""

import os
import sys

from accelerate.utils import gather_object
from trl import GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_label_utils import (
    _UNLABELED_SENTINEL,
    _extract_and_normalize,
    _majority_vote,
    grade_answer,
)


class SelfLabel4RegimeTrainer(GRPOTrainer):
    """
    Self-supervised GRPO trainer that exposes the full per-prompt rollout
    answer list to the reward function via `inputs[i]["self_answers"]`.

    Differs from `SelfLabelingGRPOTrainer` (baseline) in three ways:
      1. Builds a length-G answer list per prompt (not just the majority).
      2. Writes `inputs[i]["self_answers"]` instead of overwriting
         `inputs[i]["solution"]`. The dataset's real `solution` is preserved
         so eval reward can compare against ground truth.
      3. Self-consistency threshold drops are encoded as `[None] * G`
         (length-preserving). The downstream `compute_4regime_reward` returns
         `(0.0, "no_valid_peer")` for all-None lists, equivalent to the
         baseline sentinel-driven 0-reward behaviour.

    Note: in un-grpo-maj the "peer" is self — the rollout answer y is always in
    the support of the self-distribution (unless extraction failed). This means
    the `unseen` regime almost never fires; the active regimes are
    `confident_positive`, `uncertain_majority`, `strong_runner_up`, and `weak`.

    Args:
        self_consistency_threshold (`float`, *optional*, defaults to `0.0`):
            Below this top-answer frequency among parseable rollouts, the
            prompt's `self_answers` payload is encoded as `[None] * G`.
        log_oracle_accuracy (`bool`, *optional*, defaults to `True`):
            Log how often the majority matches the dataset's real solution
            (metric `self_labeling/pseudo_label_matches_gt`). Diagnostic only.
    """

    def __init__(
        self,
        *args,
        self_consistency_threshold: float = 0.0,
        log_oracle_accuracy: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_consistency_threshold = self_consistency_threshold
        self.log_oracle_accuracy = log_oracle_accuracy

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # Eval-mode short-circuit (identical semantics to baseline). In eval
        # we want pass@1 accuracy against the dataset's real solution, not
        # against a self-derived distribution. Skipping the self-labeling
        # path means:
        #   1. inputs[i]["solution"] keeps its dataset value (not overwritten),
        #      and inputs[i]["self_answers"] is not written. The closure-bound
        #      reward_4regime function detects the missing self_answers and
        #      falls back to binary equality against ground truth.
        #   2. self.num_generations (=train value, e.g. 8) is not used to divide
        #      N_global — parent uses num_generations_eval (typically 1), so
        #      the divisibility assertion here would crash in eval (e.g. 4 % 8).
        #   3. self_labeling/* metrics are intentionally not logged in eval
        #      because they have no meaning without majority-vote pseudo-labels.
        if not self.model.training:
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # A prompt's N rollouts are grouped contiguously in the global batch
        # (after cross-rank concatenation), but a single rank only holds a
        # slice of that batch — its local slice length is not necessarily a
        # multiple of num_generations. We therefore all-gather the parsed
        # answers, build per-prompt answer lists globally, and each rank
        # writes back only its own slice.
        G = self.num_generations
        N_local = len(inputs)
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        N_global = N_local * world_size
        assert N_global % G == 0, (
            f"global batch {N_global} (local {N_local} x world {world_size}) "
            f"not divisible by num_generations {G}"
        )
        num_groups = N_global // G

        mode = "train"

        local_answers = [_extract_and_normalize(c) for c in completions]
        local_real_solutions = [inp.get("solution") for inp in inputs]

        if world_size > 1:
            gathered_answers = gather_object(local_answers)
            gathered_real_solutions = gather_object(local_real_solutions)
        else:
            gathered_answers = local_answers
            gathered_real_solutions = local_real_solutions
        assert len(gathered_answers) == N_global, (
            f"gather_object returned {len(gathered_answers)} items, expected {N_global}"
        )

        num_parseable = sum(a is not None for a in gathered_answers)
        num_labeled = 0
        top_freq_sum = 0.0
        num_oracle_matches = 0
        # Length-preserving per-prompt answer lists (length G each). Drops
        # under self_consistency_threshold are encoded as [None] * G so the
        # downstream reward function shape stays uniform.
        self_answer_lists_global = []

        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            group_answers = list(gathered_answers[lo:hi])  # list[str | None], length G
            label, top_freq = _majority_vote(group_answers, self.self_consistency_threshold)
            top_freq_sum += top_freq

            if label is None:
                self_answer_lists_global.append([None] * G)
            else:
                self_answer_lists_global.append(group_answers)
                num_labeled += 1
                if self.log_oracle_accuracy:
                    gt_real = gathered_real_solutions[lo]
                    if gt_real is not None and grade_answer(label, gt_real):
                        num_oracle_matches += 1

        # Expand per-prompt-group answer list into per-rollout (G copies each),
        # then take this rank's slice.
        self_expanded = []
        for ans_list in self_answer_lists_global:
            self_expanded.extend([ans_list] * G)
        my_slice = self_expanded[rank * N_local : (rank + 1) * N_local]
        for i, ans_list in enumerate(my_slice):
            inputs[i]["self_answers"] = ans_list

        metrics = self._metrics[mode]
        metrics["self_labeling/fraction_labeled"].append(num_labeled / num_groups)
        metrics["self_labeling/top_frequency_mean"].append(top_freq_sum / num_groups)
        metrics["self_labeling/parseable_fraction"].append(num_parseable / N_global)
        if self.log_oracle_accuracy:
            oracle = (num_oracle_matches / num_labeled) if num_labeled > 0 else 0.0
            metrics["self_labeling/pseudo_label_matches_gt"].append(oracle)

        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
