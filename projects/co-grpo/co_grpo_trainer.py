"""Co-learning GRPO trainer: two models generate rollouts, cross-label via majority vote."""

import copy
import time
import torch
from contextlib import contextmanager

from accelerate.utils import gather_object
from trl import GRPOTrainer

from co_label_utils import _UNLABELED_SENTINEL, _extract_and_normalize, _majority_vote


class CoGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer that trains two models simultaneously with cross-labeling.

    Each step, both models generate rollouts on the same prompts. Each model computes a majority-vote
    pseudo-label from its own rollouts, then the counterpart's pseudo-label is used as ground truth
    for reward computation. This breaks the self-consistency bias of single-model unsupervised GRPO.

    Args:
        model_b (`str`):
            Path or name of the second model.
        processing_class_b:
            Tokenizer for the second model.
        peft_config_b:
            PEFT config for the second model (or `None`).
        model_b_kwargs (`dict`, *optional*):
            Additional kwargs for `AutoModelForCausalLM.from_pretrained` of model B.
        self_consistency_threshold (`float`, *optional*, defaults to `0.0`):
            Minimum top-answer frequency for a pseudo-label to be accepted.
        log_oracle_accuracy (`bool`, *optional*, defaults to `True`):
            Log how often pseudo-labels match the real ground truth.
    """

    def __init__(
        self,
        *args,
        model_b: str = None,
        processing_class_b=None,
        peft_config_b=None,
        model_b_kwargs: dict | None = None,
        self_consistency_threshold: float = 0.0,
        log_oracle_accuracy: bool = True,
        learning_rate_b: float | None = None,
        gpu_memory_utilization_b: float = 0.35,
        **kwargs,
    ):
        # Initialize model A via parent
        super().__init__(*args, **kwargs)

        self.self_consistency_threshold = self_consistency_threshold
        self.log_oracle_accuracy = log_oracle_accuracy
        self.processing_class_b = processing_class_b
        self._model_b_name = model_b

        # ---- Build model B ----
        from transformers import AutoModelForCausalLM

        model_b_kwargs = model_b_kwargs or {}
        if hasattr(self.args, "model_init_kwargs") and self.args.model_init_kwargs:
            for k, v in self.args.model_init_kwargs.items():
                model_b_kwargs.setdefault(k, v)

        self.model_b_raw = AutoModelForCausalLM.from_pretrained(model_b, **model_b_kwargs)

        # Apply PEFT if configured
        if peft_config_b is not None:
            from peft import get_peft_model
            self.model_b_raw = get_peft_model(self.model_b_raw, peft_config_b)
            if self.args.gradient_checkpointing:
                self.model_b_raw.enable_input_require_grads()

        # Enable gradient checkpointing on model B
        if self.args.gradient_checkpointing:
            gc_kwargs = self.args.gradient_checkpointing_kwargs or {}
            gc_kwargs.setdefault("use_reentrant", False)
            self.model_b_raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

        # Build vLLM engine for B BEFORE DeepSpeed wrapping (needs raw model with name_or_path)
        self.vllm_generation_b = None
        if self.use_vllm and hasattr(self, "vllm_generation") and self.vllm_generation is not None:
            from trl.generation.vllm_generation import VLLMGeneration
            vg_a = self.vllm_generation
            self.vllm_generation_b = VLLMGeneration(
                model=self.model_b_raw,
                accelerator=self.accelerator,
                is_fsdp_enabled=vg_a.is_fsdp_enabled,
                processing_class=self.processing_class_b,
                mode=vg_a.mode,
                gpu_memory_utilization=gpu_memory_utilization_b,
                max_model_length=vg_a.max_model_length,
                max_num_seqs=vg_a.max_num_seqs,
                enable_sleep_mode=vg_a.enable_sleep_mode,
                temperature=vg_a.temperature,
                top_p=vg_a.top_p,
                top_k=vg_a.top_k,
                min_p=vg_a.min_p,
                max_completion_length=vg_a.max_completion_length,
                repetition_penalty=vg_a.repetition_penalty,
            )

        # ---- Initialize model B with DeepSpeed directly ----
        # We use deepspeed.initialize instead of accelerator.prepare to avoid conflicts
        # with the accelerator's internal tracking of the "main" DeepSpeed engine.
        import deepspeed
        import json

        lr_b = learning_rate_b or self.args.learning_rate

        # Build DeepSpeed config from the accelerator's config (same ZeRO stage, etc.)
        ds_config = self.accelerator.state.deepspeed_plugin.deepspeed_config
        # Override optimizer config for model B
        ds_config_b = copy.deepcopy(ds_config)
        ds_config_b["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": lr_b,
                "weight_decay": self.args.weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        }
        ds_config_b["scheduler"] = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr_b,
                "warmup_num_steps": self.args.warmup_steps,
            },
        }
        # Set gradient accumulation and train batch size
        ds_config_b["gradient_accumulation_steps"] = self.args.gradient_accumulation_steps
        ds_config_b["train_micro_batch_size_per_gpu"] = self.args.per_device_train_batch_size
        ds_config_b["train_batch_size"] = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.accelerator.num_processes
        )

        self.model_b_engine, self.optimizer_b, _, self.lr_scheduler_b = deepspeed.initialize(
            model=self.model_b_raw,
            config=ds_config_b,
        )
        self.model_b = self.model_b_engine

        # Expose gradient checkpointing attributes through the DeepSpeed wrapper
        unwrapped_b = self.model_b_engine.module
        self.model_b.is_gradient_checkpointing = getattr(unwrapped_b, "is_gradient_checkpointing", False)

        def _gc_disable_b():
            unwrapped_b.gradient_checkpointing_disable()
            self.model_b.is_gradient_checkpointing = False

        def _gc_enable_b(gradient_checkpointing_kwargs=None):
            unwrapped_b.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            self.model_b.is_gradient_checkpointing = True

        self.model_b.gradient_checkpointing_disable = _gc_disable_b
        self.model_b.gradient_checkpointing_enable = _gc_enable_b

        self._last_loaded_step_b = -1
        self._cached_generation = None
        self._buffered_inputs_b = None

    @contextmanager
    def _as_model_b(self):
        """Context manager that temporarily swaps self.model/processing_class/vllm_generation to B's variants."""
        saved_model = self.model
        saved_processing_class = self.processing_class
        saved_vllm = getattr(self, "vllm_generation", None)
        saved_last_loaded = getattr(self, "_last_loaded_step", -1)

        self.model = self.model_b
        self.processing_class = self.processing_class_b
        if saved_vllm is not None:
            self.vllm_generation = self.vllm_generation_b
        self._last_loaded_step = self._last_loaded_step_b
        try:
            yield
        finally:
            self._last_loaded_step_b = getattr(self, "_last_loaded_step", -1)
            self.model = saved_model
            self.processing_class = saved_processing_class
            if saved_vllm is not None:
                self.vllm_generation = saved_vllm
            self._last_loaded_step = saved_last_loaded

    def _generate(self, prompts):
        """Override to support cached generation results for cross-labeling."""
        if self._cached_generation is not None:
            return self._cached_generation
        return super()._generate(prompts)

    def _compute_cross_labels(self, completions_a, completions_b, inputs):
        """Compute majority-vote pseudo-labels for each model and return cross-labels."""
        from co_label_utils import normalize_answer

        G = self.num_generations
        mode = "train" if self.model.training else "eval"
        N_local = len(completions_a)
        world_size = self.accelerator.num_processes
        N_global = N_local * world_size
        assert N_global % G == 0
        num_groups = N_global // G

        local_answers_a = [_extract_and_normalize(c) for c in completions_a]
        local_answers_b = [_extract_and_normalize(c) for c in completions_b]
        local_real_solutions = [inp.get("solution") for inp in inputs]

        if world_size > 1:
            gathered_answers_a = gather_object(local_answers_a)
            gathered_answers_b = gather_object(local_answers_b)
            gathered_real_solutions = gather_object(local_real_solutions)
        else:
            gathered_answers_a = local_answers_a
            gathered_answers_b = local_answers_b
            gathered_real_solutions = local_real_solutions

        pseudo_labels_a = []
        pseudo_labels_b = []
        num_parseable_a = sum(a is not None for a in gathered_answers_a)
        num_parseable_b = sum(a is not None for a in gathered_answers_b)
        num_labeled_a = num_labeled_b = 0
        num_oracle_matches_a = num_oracle_matches_b = 0
        num_peer_agree = 0

        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            label_a, _ = _majority_vote(gathered_answers_a[lo:hi], self.self_consistency_threshold)
            label_b, _ = _majority_vote(gathered_answers_b[lo:hi], self.self_consistency_threshold)

            pseudo_a = _UNLABELED_SENTINEL if label_a is None else label_a
            pseudo_b = _UNLABELED_SENTINEL if label_b is None else label_b

            if label_a is not None:
                num_labeled_a += 1
                if self.log_oracle_accuracy:
                    gt = normalize_answer(gathered_real_solutions[lo])
                    if gt is not None and gt == label_a:
                        num_oracle_matches_a += 1

            if label_b is not None:
                num_labeled_b += 1
                if self.log_oracle_accuracy:
                    gt = normalize_answer(gathered_real_solutions[lo])
                    if gt is not None and gt == label_b:
                        num_oracle_matches_b += 1

            if label_a is not None and label_b is not None and label_a == label_b:
                num_peer_agree += 1

            pseudo_labels_a.append(pseudo_a)
            pseudo_labels_b.append(pseudo_b)

        metrics = self._metrics[mode]
        metrics["co_labeling/labeled_fraction_a"].append(num_labeled_a / num_groups)
        metrics["co_labeling/labeled_fraction_b"].append(num_labeled_b / num_groups)
        metrics["co_labeling/parseable_fraction_a"].append(num_parseable_a / N_global)
        metrics["co_labeling/parseable_fraction_b"].append(num_parseable_b / N_global)
        both_labeled = sum(
            1 for a, b in zip(pseudo_labels_a, pseudo_labels_b)
            if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL
        )
        metrics["co_labeling/peer_agreement"].append(
            num_peer_agree / both_labeled if both_labeled > 0 else 0.0
        )
        if self.log_oracle_accuracy:
            metrics["co_labeling/oracle_accuracy_a"].append(
                num_oracle_matches_a / num_labeled_a if num_labeled_a > 0 else 0.0
            )
            metrics["co_labeling/oracle_accuracy_b"].append(
                num_oracle_matches_b / num_labeled_b if num_labeled_b > 0 else 0.0
            )

        return pseudo_labels_b, pseudo_labels_a

    def _inject_cross_labels(self, inputs, pseudo_labels, num_generations):
        """Overwrite inputs[i]["solution"] with the cross-labeled pseudo-label for each group."""
        G = num_generations
        N_local = len(inputs)
        rank = self.accelerator.process_index
        labels_expanded = []
        for label in pseudo_labels:
            labels_expanded.extend([label] * G)
        my_slice = labels_expanded[rank * N_local : (rank + 1) * N_local]
        for i, label in enumerate(my_slice):
            inputs[i]["solution"] = label

    def _co_generate_and_score(self, inputs):
        """Generate with both models, compute cross-labels, score each branch."""
        prompts = [x["prompt"] for x in inputs]

        # --- Step 1: Generate with model A ---
        gen_result_a = super()._generate(prompts)
        completions_a = gen_result_a[3]

        # --- Step 2: Generate with model B ---
        with self._as_model_b():
            gen_result_b = super()._generate(prompts)
        completions_b = gen_result_b[3]

        # --- Step 3: Compute cross-labels ---
        labels_for_a, labels_for_b = self._compute_cross_labels(
            completions_a, completions_b, inputs
        )

        # --- Step 4: Score model A's completions against B's pseudo-labels ---
        inputs_a = copy.deepcopy(inputs)
        self._inject_cross_labels(inputs_a, labels_for_a, self.num_generations)
        self._cached_generation = gen_result_a
        output_a = super()._generate_and_score_completions(inputs_a)
        self._cached_generation = None

        # --- Step 5: Score model B's completions against A's pseudo-labels ---
        inputs_b = copy.deepcopy(inputs)
        self._inject_cross_labels(inputs_b, labels_for_b, self.num_generations)
        self._cached_generation = gen_result_b
        with self._as_model_b():
            output_b = super()._generate_and_score_completions(inputs_b)
        self._cached_generation = None

        return output_a, output_b

    def _prepare_inputs(self, generation_batch):
        """Override to handle dual-model generation and buffering."""
        from trl.trainer.grpo_trainer import (
            shuffle_sequence_dict,
            split_pixel_values_by_grid,
            split_tensor_dict,
            unsplit_pixel_values_by_grid,
        )

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                output_a, output_b = self._co_generate_and_score(generation_batch)

                output_a = split_pixel_values_by_grid(output_a)
                output_a = shuffle_sequence_dict(output_a)
                batches_a = split_tensor_dict(output_a, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(b) for b in batches_a]

                output_b = split_pixel_values_by_grid(output_b)
                output_b = shuffle_sequence_dict(output_b)
                batches_b = split_tensor_dict(output_b, self.args.steps_per_generation)
                self._buffered_inputs_b = [unsplit_pixel_values_by_grid(b) for b in batches_b]

            idx = self._step % self.args.steps_per_generation
            return {"a": self._buffered_inputs[idx], "b": self._buffered_inputs_b[idx]}
        else:
            output_a, output_b = self._co_generate_and_score(generation_batch)
            return {"a": output_a, "b": output_b}

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to compute loss and backward for both models via their DeepSpeed engines."""
        time_before = time.perf_counter()

        inputs = self._prepare_inputs(inputs)
        inputs_a = inputs["a"]
        inputs_b = inputs["b"]

        # Model A: forward + backward via accelerator (routes through model A's DS engine)
        loss_a = self.compute_loss(self.model, inputs_a, num_items_in_batch=inputs_a.get("num_items_in_batch"))
        self.accelerator.backward(loss_a)

        # Model B: forward + backward via model B's own DS engine directly
        with self._as_model_b():
            loss_b = self.compute_loss(self.model, inputs_b, num_items_in_batch=inputs_b.get("num_items_in_batch"))
        # Use model B's DeepSpeed engine for backward + step
        self.model_b_engine.backward(loss_b)
        self.model_b_engine.step()

        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0

        return loss_a.detach() / self.current_gradient_accumulation_steps

    def save_model(self, output_dir=None, _internal_call=False):
        """Save both models."""
        from pathlib import Path

        output_dir = output_dir or self.args.output_dir
        output_dir = Path(output_dir)

        super().save_model(str(output_dir / "model_a"), _internal_call=_internal_call)

        model_b_unwrapped = self.model_b_engine.module
        model_b_unwrapped.save_pretrained(str(output_dir / "model_b"))
        if self.processing_class_b is not None:
            self.processing_class_b.save_pretrained(str(output_dir / "model_b"))
