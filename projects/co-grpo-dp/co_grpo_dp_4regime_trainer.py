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

import torch

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
        reward_type: str = "4regime",
        self_consistency_threshold: float = 0.0,
        log_oracle_accuracy: bool = True,
        log_distill_epsilon: float = 1e-10,
        **kwargs,
    ):
        assert my_group_name in ("A", "B"), f"my_group_name must be 'A' or 'B', got {my_group_name!r}"
        super().__init__(*args, **kwargs)
        self.my_group_name = my_group_name
        self.rendezvous = rendezvous
        self.reward_type = reward_type
        self.self_consistency_threshold = self_consistency_threshold
        self.log_oracle_accuracy = log_oracle_accuracy
        self.log_distill_epsilon = log_distill_epsilon
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

        # ---- Log-Distillation branch (cross-supervision via peer log-likelihood) ----
        # Distinct payload (token sequences + per-rollout sum logp scalars) and
        # distinct rendezvous flow (two exchanges per gen step). See
        # `_calculate_rewards_log_distill` below.
        if self.reward_type == "log_distill":
            return self._calculate_rewards_log_distill(
                inputs, prompts, completions, completion_ids_list
            )

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

    # =========================================================================
    # Log-Distillation branch (cross-supervision via peer log-likelihood)
    # =========================================================================
    #
    # Reward for rollout y_i (mine, group ME): r(y_i) = (1/T) * sum_t log p_PEER(y_{i,t} | q).
    # Two rendezvous exchanges per generation step:
    #   - mode='train_tokens': swap token sequences. ME sends my rollout tokens
    #     globally; receives peer's rollout tokens (same global ordering).
    #   - mode='train_logps':  swap per-rollout sum-logp scalars. Each group
    #     forwards peer's tokens through OWN model and sends back the per-rollout
    #     sum_t log p_ME(peer_y); the value each group RECEIVES is
    #     "peer's view of my own rollouts" = the reward signal for my rollouts.
    #
    # Both groups see identical prompts at every step (RepeatSampler / same seed),
    # so peer_tokens_global[i] aligns with the same prompt as gathered_my_tokens[i].

    def _calculate_rewards_log_distill(self, inputs, prompts, completions, completion_ids_list):
        from accelerate.utils import broadcast_object_list, gather_object

        G = self.num_generations
        N_local = len(inputs)
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        N_global = N_local * world_size
        assert N_global % G == 0, (
            f"global batch {N_global} (local {N_local} x world {world_size}) "
            f"not divisible by num_generations {G}"
        )
        mode = "train"

        # ---- 1. Decode my own completion tokens to TEXT and gather globally ----
        # Cross-family co-learning: peer's token IDs are in PEER's vocabulary
        # which may exceed MY vocab_size (e.g. Qwen vocab 152k vs Llama 128k →
        # Qwen's IDs >128k OOB in Llama's embedding/gather → CUDA assert).
        # Send decoded TEXT and let each side re-tokenize with own tokenizer.
        # This is the semantically correct level: the reward signals "how
        # likely does PEER think A's reasoning is, under PEER's own tokenization".
        local_my_text = [
            self.processing_class.decode(t, skip_special_tokens=True)
            for t in completion_ids_list
        ]
        if world_size > 1:
            gathered_my_text = gather_object(local_my_text)
        else:
            gathered_my_text = local_my_text
        assert len(gathered_my_text) == N_global, (
            f"gather_object returned {len(gathered_my_text)} text strings, expected {N_global}"
        )

        # ---- 2. Rendezvous Exchange A: swap text strings with peer ----
        # Reuse the 4regime counter (one generation step = one increment), but
        # use a distinct mode string to keep the two log_distill exchanges separate.
        gc = self._gen_counter_train
        self._gen_counter_train += 1

        if self.accelerator.is_main_process:
            peer_text_global = self.rendezvous.exchange(
                mode="train_text", counter=gc, payload=gathered_my_text
            )
            if len(peer_text_global) != N_global:
                raise RuntimeError(
                    f"peer sent {len(peer_text_global)} text strings for train_text "
                    f"gc={gc}, expected {N_global} — groups out of sync"
                )
            object_list = [peer_text_global]
        else:
            object_list = [None]
        broadcast_object_list(object_list, from_process=0)
        peer_text_global = object_list[0]

        # ---- 3. Forward peer's TEXT (re-tokenized with my tokenizer) through MY model ----
        # Each rank handles its own slice: peer_text_global[rank*N_local : (rank+1)*N_local]
        # is aligned with my local `prompts` (cross-group sampler is identical).
        my_peer_text_slice = peer_text_global[rank * N_local : (rank + 1) * N_local]
        my_logp_of_peer = self._forward_peer_logp(prompts, my_peer_text_slice)
        # my_logp_of_peer[i] = sum_t log p_ME(re-tokenized peer_text[i,t] | my prompt[i])

        # Gather across ranks → list[float|None] of length N_global (one scalar
        # per rollout in the global view, in the same global ordering as the
        # rendezvous payload).
        if world_size > 1:
            gathered_my_logp_of_peer = gather_object(my_logp_of_peer)
        else:
            gathered_my_logp_of_peer = my_logp_of_peer
        assert len(gathered_my_logp_of_peer) == N_global

        # ---- 4. Rendezvous Exchange B: swap sum-logp scalars with peer ----
        # I send "log p_ME(peer's y)" — peer uses these as rewards for THEIR rollouts.
        # I receive "log p_PEER(my y)" — I use these as rewards for MY rollouts.
        if self.accelerator.is_main_process:
            peer_logp_of_my_global = self.rendezvous.exchange(
                mode="train_logps", counter=gc, payload=gathered_my_logp_of_peer
            )
            if len(peer_logp_of_my_global) != N_global:
                raise RuntimeError(
                    f"peer sent {len(peer_logp_of_my_global)} logp scalars for train_logps "
                    f"gc={gc}, expected {N_global}"
                )
            object_list = [peer_logp_of_my_global]
        else:
            object_list = [None]
        broadcast_object_list(object_list, from_process=0)
        peer_logp_of_my_global = object_list[0]

        # ---- 5. Slice peer's reply for my rollouts, inject into inputs[i] ----
        my_peer_logp_slice = peer_logp_of_my_global[rank * N_local : (rank + 1) * N_local]
        for i in range(N_local):
            inputs[i]["peer_log_prob_sum"] = my_peer_logp_slice[i]
            inputs[i]["completion_lens"] = len(completion_ids_list[i])

        # ---- 6. Diagnostic metrics ----
        metrics = self._metrics[mode]
        n_fallback = sum(1 for v in my_peer_logp_slice if v is None)
        metrics["log_distill/fraction_fallback_in_slice"].append(n_fallback / max(N_local, 1))
        valid = [float(v) for v in my_peer_logp_slice if v is not None]
        valid_lens = [
            len(completion_ids_list[i])
            for i in range(N_local)
            if my_peer_logp_slice[i] is not None
        ]
        if valid:
            metrics["log_distill/avg_peer_logp_sum"].append(sum(valid) / len(valid))
        if valid_lens and all(L > 0 for L in valid_lens):
            per_token = [v / L for v, L in zip(valid, valid_lens)]
            metrics["log_distill/avg_per_token_logp"].append(sum(per_token) / len(per_token))

        # ---- 7. Delegate to parent for the actual reward function call ----
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    def _forward_peer_logp(self, prompts, peer_texts):
        """Forward (my_prompt + re-tokenized peer_text) sequences through MY model
        and return per-rollout sum-logp of completion positions.

        Cross-family co-learning correctness: peer's token IDs are in PEER's
        vocabulary; passing them through MY embedding would OOB when peer's
        vocab is larger (e.g. Qwen→Llama: Qwen IDs >128k OOB in Llama).
        Instead we exchange decoded TEXT and re-tokenize on the receiving side
        with the local tokenizer. Reward signal becomes "how likely does PEER
        think A's reasoning is, under PEER's own tokenization".

        Args:
            prompts: list of prompt (str OR list[dict] chat messages), length N_local.
            peer_texts: list of peer's completion TEXT (skip_special_tokens=True
                decoded by peer), length N_local.

        Returns:
            list[float | None]: per-rollout sum_t log p_ME(re-tok(peer_text)_t | q).
            None on per-row failure or empty peer completion. The log_distill
            reward closure converts None to log(epsilon) fallback.
        """
        tokenizer = self.processing_class
        device = next(self.model.parameters()).device
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        if not prompts or not peer_texts:
            return []
        if len(prompts) != len(peer_texts):
            raise RuntimeError(
                f"prompt count {len(prompts)} != peer text count {len(peer_texts)}"
            )

        # Normalize prompts to TEXT (apply chat template if conversational, match
        # generation-time format with add_generation_prompt=True).
        first_prompt = prompts[0] if prompts else None
        if isinstance(first_prompt, str) or first_prompt is None:
            prompts_text = list(prompts)
        elif hasattr(tokenizer, "apply_chat_template"):
            prompts_text = [
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]
        else:
            print(f"[log_distill] cannot normalize prompts (type={type(first_prompt)!r})")
            return [None] * len(prompts)

        # Tokenize: prompt (no special — already includes chat tags) + peer's
        # text re-tokenized under MY tokenizer.
        try:
            prompt_encs = tokenizer(prompts_text, add_special_tokens=False)
            prompt_ids_list = prompt_encs["input_ids"]
            peer_encs = tokenizer(list(peer_texts), add_special_tokens=False)
            peer_token_ids_list = peer_encs["input_ids"]
        except Exception as e:
            print(f"[log_distill] tokenize failed: {e}")
            return [None] * len(prompts)

        B = len(prompt_ids_list)
        if B == 0:
            return []

        prompt_lens = [len(p) for p in prompt_ids_list]
        peer_lens = [len(c) for c in peer_token_ids_list]
        max_p = max(prompt_lens) if prompt_lens else 0
        max_c = max(peer_lens) if peer_lens else 0
        if max_c == 0:
            # All peer completions empty or tokenized to nothing — nothing to evaluate
            return [None] * B
        L = max_p + max_c

        input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        peer_mask = torch.zeros((B, max_c), dtype=torch.long, device=device)

        for i, (p, c) in enumerate(zip(prompt_ids_list, peer_token_ids_list)):
            p_len = len(p)
            c_len = len(c)
            if c_len == 0:
                continue  # row stays all-pad, will be marked None downstream
            input_ids[i, max_p - p_len : max_p] = torch.tensor(p, dtype=torch.long, device=device)
            attention_mask[i, max_p - p_len : max_p] = 1
            input_ids[i, max_p : max_p + c_len] = torch.tensor(c, dtype=torch.long, device=device)
            attention_mask[i, max_p : max_p + c_len] = 1
            peer_mask[i, :c_len] = 1

        # Use parent's chunked forward + selective_log_softmax (chosen-token per-position logp).
        chunk_size = max(1, getattr(self.args, "per_device_train_batch_size", 4) or 4)

        try:
            with torch.no_grad():
                logps, _ = self._get_per_token_logps_and_entropies(
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=max_c,
                    batch_size=chunk_size,
                    compute_entropy=False,
                )
            # logps shape: (B, max_c). Sum over peer_mask per rollout.
            peer_mask_float = peer_mask.float()
            sum_logp = (logps * peer_mask_float).sum(dim=-1).cpu().tolist()
        except Exception as e:
            print(f"[log_distill] forward failed: {e}")
            return [None] * B

        # Empty peer completion rows → None (peer_mask all zero, sum=0 isn't a
        # legit zero-prob signal). Otherwise return float sum-logp.
        result = []
        for i, s in enumerate(sum_logp):
            if peer_lens[i] == 0:
                result.append(None)
            else:
                result.append(float(s))
        return result
