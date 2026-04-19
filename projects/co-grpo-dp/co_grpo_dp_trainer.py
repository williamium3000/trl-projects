"""Cross-supervised GRPO with data-parallel split: each group trains one model.

Two accelerate worlds run in parallel (group A on CUDA_VISIBLE_DEVICES=0..N-1,
group B on N..2N-1). Each group is a standard `GRPOTrainer` with a single
override: `_calculate_rewards` computes this group's pseudo-labels, exchanges
them with the peer group via a file rendezvous, and injects the peer's pseudo-
labels into `inputs[i]["solution"]` before delegating to the parent reward path.

This override is the *only* coupling between the two groups. Generation,
forward, backward, and DS->vLLM weight sync all happen independently inside
each group, so the two groups run in genuine parallel across disjoint GPUs.
"""

from accelerate.utils import broadcast_object_list, gather_object
from trl import GRPOTrainer

from co_label_utils import (
    _UNLABELED_SENTINEL,
    _extract_and_normalize,
    _majority_vote,
    normalize_answer,
)


class CoGRPOdpTrainer(GRPOTrainer):
    """
    Args:
        my_group_name (`str`):
            `'A'` or `'B'`. Identifies which half of the run this process belongs to.
        rendezvous (`Rendezvous`):
            File-based communicator to the peer group. Only the main process of
            each group calls `exchange()`; the rest receive via broadcast.
        self_consistency_threshold (`float`, *optional*, defaults to `0.0`):
            Minimum top-answer frequency (over parseable rollouts per prompt group)
            for this group's pseudo-label to be accepted. `0.0` takes the plurality
            winner. Groups below the threshold are labeled with `_UNLABELED_SENTINEL`
            so the peer's accuracy reward evaluates to 0.0 for every rollout in them.
        log_oracle_accuracy (`bool`, *optional*, defaults to `True`):
            Log how often this group's pseudo-label matches the dataset's real
            `solution` (metric `co_labeling/oracle_accuracy_me`). Purely diagnostic;
            the real label never influences training.
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
        # Rendezvous counter advances once per call to `_calculate_rewards`
        # (i.e., once per generation step), NOT per training_step. `_calculate_rewards`
        # is only invoked inside `_generate_and_score_completions`, which the parent
        # calls every `steps_per_generation * num_iterations` training steps.
        # Train and eval counters are tracked separately because both modes trigger
        # this hook; sharing a single counter would misalign the two groups as soon
        # as evaluation runs.
        self._gen_counter_train = 0
        self._gen_counter_eval = 0

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # A prompt's N rollouts are grouped contiguously in the global batch (after
        # cross-rank concatenation), but a single rank only holds a slice of that
        # batch — its local slice length is not necessarily a multiple of
        # num_generations. We therefore all-gather parsed answers within our group,
        # compute pseudo-labels globally, exchange them with the peer group, and
        # each rank writes back only its own slice of the peer's pseudo-labels.
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

        # ---- 2. Majority vote my pseudo-labels over my own G rollouts per prompt ----
        my_pseudo = []
        num_labeled_me = 0
        num_oracle_me = 0
        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            label, _ = _majority_vote(gathered_answers[lo:hi], self.self_consistency_threshold)
            if label is None:
                my_pseudo.append(_UNLABELED_SENTINEL)
            else:
                my_pseudo.append(label)
                num_labeled_me += 1
                if self.log_oracle_accuracy:
                    gt = normalize_answer(gathered_real_solutions[lo])
                    if gt is not None and gt == label:
                        num_oracle_me += 1

        # ---- 3. Exchange pseudo-labels with peer group via file rendezvous ----
        # Only the main process of each group touches the filesystem; the rest
        # receive peer's pseudo-labels via in-group broadcast.
        if mode == "train":
            gc = self._gen_counter_train
            self._gen_counter_train += 1
        else:
            gc = self._gen_counter_eval
            self._gen_counter_eval += 1

        if self.accelerator.is_main_process:
            peer_pseudo = self.rendezvous.exchange(mode=mode, counter=gc, payload=my_pseudo)
            # Sanity: peer must send same number of prompt groups.
            if len(peer_pseudo) != num_groups:
                raise RuntimeError(
                    f"peer sent {len(peer_pseudo)} pseudo-labels for {mode} gc={gc}, "
                    f"expected {num_groups} — groups out of sync"
                )
            object_list = [peer_pseudo]
        else:
            object_list = [None]
        # Broadcast a single-element list containing the peer_pseudo list.
        # (broadcast_object_list modifies the list inplace.)
        broadcast_object_list(object_list, from_process=0)
        peer_pseudo = object_list[0]

        # ---- 4. Cross-labeling metrics ----
        metrics = self._metrics[mode]
        num_labeled_peer = sum(1 for p in peer_pseudo if p != _UNLABELED_SENTINEL)
        both_labeled = sum(
            1 for a, b in zip(my_pseudo, peer_pseudo)
            if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL
        )
        peer_agree = sum(
            1 for a, b in zip(my_pseudo, peer_pseudo)
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

        # ---- 5. Inject peer's pseudo-labels into this rank's local slice ----
        # Expand per-prompt-group label into per-rollout labels (G copies each),
        # then take this rank's [rank * N_local, (rank + 1) * N_local) slice.
        peer_expanded = []
        for label in peer_pseudo:
            peer_expanded.extend([label] * G)
        my_slice = peer_expanded[rank * N_local : (rank + 1) * N_local]
        for i, label in enumerate(my_slice):
            inputs[i]["solution"] = label

        # ---- 6. Delegate to parent for the actual reward function call ----
        # Parent will gather rewards_per_func across my group (not across the peer
        # group — the two groups have disjoint process groups). Group-internal
        # gather + group-internal advantage normalization is exactly what GRPO
        # semantics call for: each model normalizes its own rewards.
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
