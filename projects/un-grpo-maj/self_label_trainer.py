import os
import sys
from collections import Counter

from accelerate.utils import gather_object
from trl import GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifiers.qwen.qwen_math_parser import extract_answer
from verifiers.qwen.math_normalize import normalize_answer as _qwen_normalize
from verifiers.qwen.math_grade import grade_answer


def extract_boxed_answer(text):
    """Alias for co-grpo/co_label_utils.py importlib bridge compatibility."""
    return extract_answer(text, "math")


def normalize_answer(answer):
    if answer is None:
        return None
    return _qwen_normalize(answer)


# Sentinel written into `solution` for prompt groups that fail the self-consistency
# threshold. It cannot match any parsed \boxed{} answer, so reward_correctness
# returns 0.0 for every rollout in such a group — no reward-function change needed.
_UNLABELED_SENTINEL = "\x00__unlabeled__\x00"


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _extract_and_normalize(completion):
    result = normalize_answer(extract_answer(_get_text(completion), "math"))
    if result is None or result == "":
        return None
    return result


def _majority_vote(answers, threshold):
    """
    Args:
        answers (`list[str | None]`):
            One normalized parsed answer per rollout in the prompt group.
            `None` means the rollout did not produce a parseable `\\boxed{}`.
        threshold (`float`):
            Minimum top-answer frequency (over parseable answers) to accept
            the majority as the pseudo-label.

    Returns:
        `tuple[str | None, float]`: `(pseudo_label, top_frequency)`. `pseudo_label`
        is `None` when no rollout parses, or when the top frequency is below
        `threshold`. `top_frequency` is `0.0` when no rollout parses.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0
    counts = Counter(valid)
    top_answer, top_count = counts.most_common(1)[0]
    top_freq = top_count / len(valid)
    if top_freq < threshold:
        return None, top_freq
    return top_answer, top_freq


class SelfLabelingGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer that replaces the ground-truth `solution` with a majority-vote
    pseudo-label computed across the N rollouts of each prompt, then delegates
    reward computation to the parent class.

    Args:
        self_consistency_threshold (`float`, *optional*, defaults to `0.0`):
            Minimum fraction (over parseable rollouts) that the top answer must
            reach for a prompt group to be labeled. `0.0` accepts the plurality
            winner. When a group fails the threshold, every rollout in it receives
            a sentinel `solution` so the accuracy reward evaluates to `0.0`.
        log_oracle_accuracy (`bool`, *optional*, defaults to `True`):
            If `True`, also log how often the pseudo-label matches the real
            `solution` from the dataset (metric `self_labeling/pseudo_label_matches_gt`).
            Purely diagnostic — the real label does not influence training.
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
        # Eval-mode short-circuit. In eval we want pass@1 accuracy against the
        # dataset's real solution, not against a self-majority pseudo-label.
        # Skipping the self-labeling path means:
        #   1. inputs[i]["solution"] keeps its dataset value (not overwritten),
        #      so the parent's reward path (reward_correctness) compares the
        #      completion against ground truth via grade_answer.
        #   2. self.num_generations (=train value, e.g. 8) is not used to divide
        #      N_global — parent uses num_generations_eval (typically 1), so
        #      the divisibility assertion here would crash in eval (e.g. 4 % 8).
        #   3. self_labeling/* metrics are intentionally not logged in eval
        #      because they have no meaning without majority-vote pseudo-labels.
        if not self.model.training:
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # A prompt's N rollouts are grouped *contiguously in the global batch* (after
        # cross-rank concatenation), but a single rank only holds a slice of that batch —
        # its local slice length is not necessarily a multiple of num_generations. We
        # therefore all-gather the parsed answers, compute pseudo-labels globally, and
        # each rank writes back only its own slice.
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

        mode = "train" if self.model.training else "eval"

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
        pseudo_labels_global = []

        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            label, top_freq = _majority_vote(
                gathered_answers[lo:hi], self.self_consistency_threshold
            )
            top_freq_sum += top_freq

            if label is None:
                pseudo = _UNLABELED_SENTINEL
            else:
                pseudo = label
                num_labeled += 1
                if self.log_oracle_accuracy:
                    gt_real = gathered_real_solutions[lo]
                    if gt_real is not None and grade_answer(label, gt_real):
                        num_oracle_matches += 1

            pseudo_labels_global.extend([pseudo] * G)

        my_slice = pseudo_labels_global[rank * N_local : (rank + 1) * N_local]
        for i, pseudo in enumerate(my_slice):
            inputs[i]["solution"] = pseudo

        metrics = self._metrics[mode]
        metrics["self_labeling/fraction_labeled"].append(num_labeled / num_groups)
        metrics["self_labeling/top_frequency_mean"].append(top_freq_sum / num_groups)
        metrics["self_labeling/parseable_fraction"].append(num_parseable / N_global)
        if self.log_oracle_accuracy:
            oracle = (num_oracle_matches / num_labeled) if num_labeled > 0 else 0.0
            metrics["self_labeling/pseudo_label_matches_gt"].append(oracle)

        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
