"""Cross-supervised GRPO with data-parallel split — 4-regime confidence-gated reward.

Same mechanics as `co_grpo_dp_trainer.py` (two accelerate worlds, file rendezvous),
but the peer payload carries the full per-prompt list of canonical answers (not
just the majority). Each rollout's `inputs[i]["peer_answers"]` is filled with
the peer group's length-G answer list for that prompt, and the closure-bound
`reward_4regime` reward function in `train_co_grpo_dp_4regime.py` consumes it.

Eval mode short-circuits exactly like the baseline trainer: rendezvous is not
touched, `inputs[i]["peer_answers"]` is not written, and the reward function
falls back to binary equality against the dataset's real `inputs[i]["solution"]`.
"""

from accelerate.utils import broadcast_object_list, gather_object
from trl import GRPOTrainer

from co_label_utils import (
    _UNLABELED_SENTINEL,
    _extract_and_normalize,
    _majority_vote,
    normalize_answer,
)


class CoGRPOdp4RegimeTrainer(GRPOTrainer):
    """
    Args:
        my_group_name (`str`):
            `'A'` or `'B'`. Identifies which half of the run this process belongs to.
        rendezvous (`Rendezvous`):
            File-based communicator to the peer group. Only the main process of
            each group calls `exchange()`; the rest receive via broadcast.
        self_consistency_threshold (`float`, *optional*, defaults to `0.0`):
            Below this top-answer frequency among parseable peer rollouts, the
            peer group for that prompt is dropped (payload entry encoded as
            `[None] * num_generations`). The downstream `reward_4regime`
            function then sees an all-`None` peer and returns the
            `no_valid_peer` regime (reward 0). This mirrors the baseline
            trainer's sentinel-driven 0-reward behaviour for unlabeled groups.
        log_oracle_accuracy (`bool`, *optional*, defaults to `True`):
            Log how often this group's majority matches the dataset's real
            `solution` (metric `co_labeling/oracle_accuracy_me`). Diagnostic only.
    """

    def __init__(
        self,
        *args,
        my_group_name: str,
        rendezvous,
        self_consistency_threshold: float = 0.0,
        log_oracle_accuracy: bool = True,
        **kwargs,
    ):
        assert my_group_name in ("A", "B"), f"my_group_name must be 'A' or 'B', got {my_group_name!r}"
        super().__init__(*args, **kwargs)
        self.my_group_name = my_group_name
        self.rendezvous = rendezvous
        self.self_consistency_threshold = self_consistency_threshold
        self.log_oracle_accuracy = log_oracle_accuracy
        # Rendezvous counter advances once per call to `_calculate_rewards` in
        # train mode (i.e., once per train generation step), NOT per training
        # step. `_calculate_rewards` is only invoked inside
        # `_generate_and_score_completions`, which the parent calls every
        # `steps_per_generation * num_iterations` training steps. Eval mode
        # short-circuits before touching rendezvous, so no eval counter is needed.
        self._gen_counter_train = 0

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # Eval-mode short-circuit (identical semantics to baseline). In eval we
        # want pass@1 accuracy on the validation set against the dataset's real
        # solution, not against a peer-supplied distribution. Skipping the
        # cross-labeling path means:
        #   1. inputs[i]["solution"] keeps its dataset value (not overwritten),
        #      and inputs[i]["peer_answers"] is not written. The closure-bound
        #      reward_4regime function detects the missing peer_answers and
        #      falls back to binary equality against ground truth.
        #   2. self.rendezvous is never touched in eval, so the two groups do
        #      not need to be in lockstep during eval (one can finish first).
        #   3. self._gen_counter_train is not advanced by eval, so train-mode
        #      rendezvous alignment with the peer survives any number of eval
        #      runs interleaved between train steps.
        if not self.model.training:
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # ---- Train mode: cross-labeling + peer rendezvous ----
        # A prompt's N rollouts are grouped contiguously in the global batch
        # (after cross-rank concatenation), but a single rank only holds a
        # slice of that batch — its local slice length is not necessarily a
        # multiple of num_generations. We therefore all-gather parsed answers
        # within our group, exchange the full per-prompt answer lists with the
        # peer group, and each rank writes back only its own slice.
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

        # ---- 1. Gather my group's answers and the dataset's real solutions ----
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

        # ---- 2. Build per-prompt answer lists (with self-consistency drops) ----
        # Payload diverges from baseline: we send the full G-length answer
        # list per prompt, not just the majority. Self-consistency threshold
        # drops are encoded as [None] * G (length-preserving) so downstream
        # reward function shape stays uniform, and `compute_4regime_reward`
        # naturally returns 0 in the no_valid_peer regime for those.
        my_answer_lists = []
        my_majority_for_metrics = []
        num_labeled_me = 0
        num_oracle_me = 0
        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            group_answers = list(gathered_answers[lo:hi])  # list[str | None], length G
            label, _ = _majority_vote(group_answers, self.self_consistency_threshold)
            if label is None:
                my_answer_lists.append([None] * G)
                my_majority_for_metrics.append(_UNLABELED_SENTINEL)
            else:
                my_answer_lists.append(group_answers)
                my_majority_for_metrics.append(label)
                num_labeled_me += 1
                if self.log_oracle_accuracy:
                    gt = normalize_answer(gathered_real_solutions[lo])
                    if gt is not None and gt == label:
                        num_oracle_me += 1

        # ---- 3. Exchange answer lists with peer group via file rendezvous ----
        # Only the main process of each group touches the filesystem; the rest
        # receive peer's payload via in-group broadcast.
        # NB: only train-mode rendezvous (eval short-circuits before this).
        gc = self._gen_counter_train
        self._gen_counter_train += 1

        if self.accelerator.is_main_process:
            peer_answer_lists = self.rendezvous.exchange(
                mode=mode, counter=gc, payload=my_answer_lists
            )
            # Sanity: peer must send same number of prompt groups.
            if len(peer_answer_lists) != num_groups:
                raise RuntimeError(
                    f"peer sent {len(peer_answer_lists)} answer lists for {mode} gc={gc}, "
                    f"expected {num_groups} — groups out of sync"
                )
            object_list = [peer_answer_lists]
        else:
            object_list = [None]
        broadcast_object_list(object_list, from_process=0)
        peer_answer_lists = object_list[0]

        # ---- 4. Cross-labeling metrics (computed from majorities for compat) ----
        # Same metric definitions as baseline trainer so wandb panels remain
        # comparable across baseline and 4regime runs. Majorities are derived
        # locally from the received per-prompt answer lists.
        peer_majority = [_majority_vote(lst, 0.0)[0] for lst in peer_answer_lists]
        peer_majority_str = [m if m is not None else _UNLABELED_SENTINEL for m in peer_majority]
        metrics = self._metrics[mode]
        num_labeled_peer = sum(1 for p in peer_majority_str if p != _UNLABELED_SENTINEL)
        both_labeled = sum(
            1 for a, b in zip(my_majority_for_metrics, peer_majority_str)
            if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL
        )
        peer_agree = sum(
            1 for a, b in zip(my_majority_for_metrics, peer_majority_str)
            if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL and a == b
        )
        metrics["co_labeling/peer_agreement"].append(
            peer_agree / both_labeled if both_labeled > 0 else 0.0
        )
        metrics["co_labeling/labeled_fraction_me"].append(num_labeled_me / num_groups)
        metrics["co_labeling/labeled_fraction_peer"].append(num_labeled_peer / num_groups)
        metrics["co_labeling/both_labeled_fraction"].append(both_labeled / num_groups)
        if self.log_oracle_accuracy:
            metrics["co_labeling/oracle_accuracy_me"].append(
                num_oracle_me / num_labeled_me if num_labeled_me > 0 else 0.0
            )

        # ---- 5. Inject peer's per-prompt answer list into this rank's local slice ----
        # Every rollout in a prompt group shares the same peer distribution
        # (G copies of the same list).
        peer_expanded = []
        for ans_list in peer_answer_lists:
            peer_expanded.extend([ans_list] * G)
        my_slice = peer_expanded[rank * N_local : (rank + 1) * N_local]
        for i, ans_list in enumerate(my_slice):
            inputs[i]["peer_answers"] = ans_list

        # ---- 6. Delegate to parent for the actual reward function call ----
        # Parent will gather inputs[i]["peer_answers"] into reward_kwargs (per
        # `_calculate_rewards` keys logic) and pass it to the closure-bound
        # reward_4regime function.
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
