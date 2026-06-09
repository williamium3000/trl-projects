# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""co-OPSD trainer: two-model on-policy co-distillation.

OPSD is single-model self-distillation: one model is both student and teacher,
the teacher merely sees the ground-truth solution and is *not* updated.

co-OPSD makes the relation symmetric and mutual. Each step:

  - model1 samples an on-policy trajectory ``traj1``; model2 scores it and
    ``loss1`` updates **model1** (model2's logits are detached).
  - model2 samples an on-policy trajectory ``traj2``; model1 scores it and
    ``loss2`` updates **model2** (model1's logits are detached).

Both models are held by a single :class:`CoModelPair` ``nn.Module`` so one HF
``Trainer`` / one optimizer drives them; ``CoModelPair.forward`` runs all four
sub-forwards and returns ``loss1 + loss2``, which keeps DDP gradient
all-reduce correct for both models. The cross-detach guarantees a single
backward updates each model only from its own student loss.

**Loss.** For a same-tokenizer model pair the per-direction loss is the exact
generalized JSD (copied from OPSD). For a cross-tokenizer pair the trajectory
is decoded and re-encoded with the scoring model's tokenizer, and the loss is
GOLD (the HuggingFaceH4 "on-policy distillation" method) —
`trl.experimental.gold` `ULDLoss` with `use_extended_uld=True` (token-merging
sequence alignment) and `uld_use_hybrid_loss=True`: `L_GOLD = w1*L_GKD +
w2*L_ULD`, i.e. exact GKD/JSD on tokens shared by both vocabularies and the ULD
sorted-probability distance only on the unmatched remainder.

**Generation.** With `use_vllm` each process holds one colocated vLLM engine
per model; trajectories are sampled there and both models' updated weights are
synced into their engines after every optimizer step. Otherwise HF `.generate()`
is used (correct but ~5x slower).
"""

import os
import random
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer_callback import TrainerCallback

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import disable_dropout_in_model, ensure_master_addr_port
from trl.experimental.gold.gold_trainer import ULDLoss

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = SamplingParams = None


class CoVLLMSyncCallback(TrainerCallback):
    """After each optimizer step, sync both models' updated weights into their
    colocated vLLM engines so the next step generates on-policy."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer.use_vllm and self.trainer.accelerator.sync_gradients:
            self.trainer._sync_vllm()


class EMATeacherUpdateCallback(TrainerCallback):
    """After each optimizer step, refresh both peers' EMA teacher weights
    (ema = decay*ema + (1-decay)*weight). First call lazily initializes the EMA."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer.accelerator.unwrap_model(self.trainer.model).update_ema()


def generalized_jsd_loss(
    student_logits,
    teacher_logits,
    labels=None,
    beta=0.5,
    temperature=1.0,
    token_clip=None,
    reduction="batchmean",
):
    """Generalized Jensen-Shannon Divergence loss.

    Copied verbatim from OPSD's ``OPSDTrainer.generalized_jsd_loss`` (consistency
    over abstraction — see the repo AGENTS.md). ``beta=0`` reduces to the forward
    KL ``KL(teacher || student)``; ``token_clip`` caps each token's divergence so
    high-divergence stylistic tokens do not dominate the gradient.

    Args:
        student_logits (`torch.Tensor`):
            Student logits over the trajectory tokens, shape `[batch, seq, vocab]`.
        teacher_logits (`torch.Tensor`):
            Teacher logits over the same trajectory tokens.
        labels (`torch.Tensor`, *optional*):
            Per-token labels; positions equal to `-100` are excluded from the loss.
        beta (`float`, *optional*, defaults to `0.5`):
            Interpolation factor of the generalized JSD.
        temperature (`float`, *optional*, defaults to `1.0`):
            Softmax temperature applied to both logits.
        token_clip (`float`, *optional*):
            If set, clips each token's divergence to this maximum before reduction.
        reduction (`str`, *optional*, defaults to `"batchmean"`):
            Reduction mode.
    """
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta * kl_teacher + (1 - beta) * kl_student

    # Per-token clipping: cap each token's divergence value.
    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)

    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    if reduction == "batchmean":
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    else:
        return jsd


class CoModelPair(nn.Module):
    """Holds the two co-OPSD models so a single ``Trainer`` / optimizer drives both.

    ``forward`` consumes the trajectory-augmented inputs prepared by
    :meth:`CoOPSDTrainer.training_step` and returns ``{"loss": loss1 + loss2}``.
    For each direction the teacher forward runs under ``torch.no_grad`` so the
    teacher model receives no gradient from the other model's student loss.

    When ``uld_dir1`` / ``uld_dir2`` are provided the corresponding direction
    uses GOLD's cross-tokenizer ULD loss; otherwise it uses the exact JSD.
    """

    def __init__(self, model1, model2, beta=0.0, temperature=1.0, jsd_token_clip=None,
                 use_ema_teacher=False, ema_decay=0.999):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.beta = beta
        self.temperature = temperature
        self.jsd_token_clip = jsd_token_clip
        # GOLD ULD loss per direction; assigned by CoOPSDTrainer once the
        # accelerator device is known. None => exact same-tokenizer JSD.
        self.uld_dir1 = None
        self.uld_dir2 = None
        # Trainer reaches for `.config`; expose model1's so its checks succeed.
        self.config = model1.config
        # EMA teacher: each peer scores the other's trajectory with a slow-moving
        # EMA of its own trainable (LoRA) weights instead of its live weights. This
        # restores the "stable teacher" anchor that single-model OPSD's
        # `fixed_teacher` provides — which a live, co-trained peer otherwise lacks —
        # to curb the moving-target drift that collapses homogeneous co-OPSD. The
        # EMA dicts stay None until the first optimizer step; until then the swap is
        # a no-op and the live peer weights are used. Assumes DeepSpeed ZeRO-2
        # (params unsharded per rank), matching `_sync_vllm`'s assumption.
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema1 = None  # EMA of model1's trainable params (model1-as-teacher, dir2)
        self._ema2 = None  # EMA of model2's trainable params (model2-as-teacher, dir1)

    @property
    def is_gradient_checkpointing(self):
        # Both models share the same setting (toggled together below).
        return self.model1.is_gradient_checkpointing

    def gradient_checkpointing_enable(self, **kwargs):
        self.model1.gradient_checkpointing_enable(**kwargs)
        self.model2.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.model1.gradient_checkpointing_disable()
        self.model2.gradient_checkpointing_disable()

    @contextmanager
    def _ema_swap(self, model, ema_params):
        """Temporarily load `ema_params` into `model`'s trainable weights for a
        teacher forward, then restore the live weights. No-op when EMA is disabled
        or not yet initialized (falls back to live peer weights). Safe inside
        `torch.no_grad()`. ZeRO-2 only (params are full per rank), matching the
        rest of this trainer; ZeRO-3 would need a `GatheredParameters` wrap.
        """
        if not self.use_ema_teacher or ema_params is None:
            yield
            return
        saved = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in ema_params:
                continue
            ema = ema_params[name]
            if ema.device != param.data.device:
                ema = ema.to(param.data.device)
                ema_params[name] = ema
            saved[name] = param.data.clone()
            param.data.copy_(ema)
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if name in saved:
                    param.data.copy_(saved[name])

    def _update_one_ema(self, model, ema_params):
        """`ema = decay*ema + (1-decay)*weight` over trainable params. The first
        call (ema_params is None) snapshots the current weights and returns them
        as the initial EMA (no decay step yet)."""
        if ema_params is None:
            return {name: p.data.clone().detach()
                    for name, p in model.named_parameters() if p.requires_grad}
        decay = self.ema_decay
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in ema_params:
                continue
            ema = ema_params[name]
            if ema.device != param.data.device:
                ema = ema.to(param.data.device)
                ema_params[name] = ema
            ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
        return ema_params

    def update_ema(self):
        """Refresh both peers' EMA after an optimizer step (lazy-init on first call)."""
        if not self.use_ema_teacher:
            return
        self._ema1 = self._update_one_ema(self.model1, self._ema1)
        self._ema2 = self._update_one_ema(self.model2, self._ema2)

    def _distill_one_direction(self, student_model, teacher_model, batch, prefix, uld_loss, teacher_ema=None):
        """Compute one direction's loss: ``student_model`` learns from ``teacher_model``.

        ``batch[prefix + ...]`` carries the student full sequence, the teacher
        full sequence, their prompt lengths, and both sets of labels (prompt and
        padding masked to `-100`). With ``uld_loss`` the cross-tokenizer ULD loss
        is used; otherwise the exact generalized JSD.
        """
        s_ids = batch[f"{prefix}_student_ids"]
        s_mask = batch[f"{prefix}_student_mask"]
        s_prompt_len = batch[f"{prefix}_student_prompt_len"]
        s_labels = batch[f"{prefix}_student_labels"]
        t_ids = batch[f"{prefix}_teacher_ids"]
        t_mask = batch[f"{prefix}_teacher_mask"]
        t_prompt_len = batch[f"{prefix}_teacher_prompt_len"]
        t_labels = batch[f"{prefix}_teacher_labels"]

        # Student forward (with gradients).
        student_out = student_model(input_ids=s_ids, attention_mask=s_mask)
        # Teacher forward (no gradients): the teacher is a detached scorer here.
        # `_ema_swap` is a no-op unless EMA teacher is on, in which case the peer
        # scores with its slow-moving EMA weights (restored before its own backward).
        with torch.no_grad(), self._ema_swap(teacher_model, teacher_ema):
            teacher_out = teacher_model(input_ids=t_ids, attention_mask=t_mask)

        if uld_loss is not None:
            # GOLD's ULD: pass full logits / labels / ids; it extracts the answer
            # region from the labels and aligns the two tokenizations internally.
            return uld_loss(
                student_logits=student_out.logits,
                teacher_logits=teacher_out.logits,
                student_labels=s_labels,
                teacher_labels=t_labels,
                student_input_ids=s_ids,
                teacher_input_ids=t_ids,
            )

        # Same-tokenizer path: exact JSD over the (shared) trajectory tokens.
        student_logits = student_out.logits[:, s_prompt_len - 1 : -1, :]
        teacher_logits = teacher_out.logits[:, t_prompt_len - 1 : -1, :]
        traj_labels = s_labels[:, s_prompt_len:]
        return generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=traj_labels,
            beta=self.beta,
            temperature=self.temperature,
            token_clip=self.jsd_token_clip,
        )

    def forward(self, **inputs):
        # Direction 1: model1 is student, model2 scores model1's trajectory (EMA of model2).
        loss1 = self._distill_one_direction(self.model1, self.model2, inputs, "dir1", self.uld_dir1, teacher_ema=self._ema2)
        # Direction 2: model2 is student, model1 scores model2's trajectory (EMA of model1).
        loss2 = self._distill_one_direction(self.model2, self.model1, inputs, "dir2", self.uld_dir2, teacher_ema=self._ema1)
        return {"loss": loss1 + loss2, "loss1": loss1.detach(), "loss2": loss2.detach()}


class CoOPSDTrainer(Trainer):
    """Two-model on-policy co-distillation trainer.

    Args:
        model1, model2 (`PreTrainedModel`):
            The two models that co-distill into each other.
        args (`GOLDConfig`):
            Training configuration (reuses OPSD's `GOLDConfig`).
        train_dataset (`PairedDataset`):
            Paired stream from `co_opsd_data.build_paired_dataset`.
        data_collator (`CoSelfDistillationDataCollator`):
            Builds both models' student/teacher prompts.
        tokenizer1, tokenizer2 (`PreTrainedTokenizerBase`):
            Each model's tokenizer.
        teacher_sees_gt_answer (`bool`, *optional*, defaults to `True`):
            Whether the scoring (teacher) prompt embeds the ground-truth solution.
        distill_loss_type (`str`, *optional*, defaults to `"auto"`):
            Per-direction distillation loss, one of:

            - `"jsd"`: exact generalized JSD (requires a shared vocabulary).
            - `"uld"`: original Universal Logit Distillation — positional
              truncation + sorted-probability L1 distance.
            - `"gold"`: GOLD (HuggingFaceH4) — token-merging alignment + the
              hybrid loss `L_GOLD = w1*L_GKD + w2*L_ULD` (exact JSD on tokens
              shared by both vocabularies, ULD sorting on the rest).
            - `"auto"`: `"jsd"` for a same-tokenizer pair, `"gold"` otherwise.

            A same-tokenizer pair always uses `"jsd"` regardless of this value,
            since ULD/GOLD only make sense across different vocabularies.
    """

    def __init__(
        self,
        model1,
        model2,
        args,
        train_dataset,
        data_collator,
        tokenizer1,
        tokenizer2,
        teacher_sees_gt_answer=True,
        distill_loss_type="auto",
        use_ema_teacher=False,
        ema_decay=0.999,
        callbacks=None,
    ):
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.teacher_sees_gt_answer = teacher_sees_gt_answer
        self.use_ema_teacher = use_ema_teacher
        # Per-direction losses accumulated between `log` calls (see `log`).
        self._loss1_log = []
        self._loss2_log = []

        # Resolve the per-direction loss. A shared vocabulary => exact JSD; ULD
        # and GOLD only make sense across different tokenizers.
        same_vocab = tokenizer1.get_vocab() == tokenizer2.get_vocab()
        if same_vocab:
            if distill_loss_type not in ("auto", "jsd"):
                print(f"[co-OPSD] same-tokenizer pair — overriding "
                      f"distill_loss_type='{distill_loss_type}' with 'jsd'.")
            loss_mode = "jsd"
        elif distill_loss_type == "auto":
            loss_mode = "gold"
        else:
            loss_mode = distill_loss_type
        self.loss_mode = loss_mode
        self.cross_tokenizer = loss_mode in ("uld", "gold")
        print(f"[co-OPSD] distillation loss: {loss_mode}")

        # The collated batch carries co-OPSD-specific keys that do not match any
        # model `forward` signature; keep them instead of letting Trainer drop them.
        args.remove_unused_columns = False

        # uld  = original ULD: positional truncation + sorted-probability distance.
        # gold = GOLD (HuggingFaceH4): extended-ULD token-merging alignment + the
        #        hybrid loss L_GOLD = w1*L_GKD + w2*L_ULD (exact JSD on tokens
        #        shared by both vocabularies, ULD sorting only on the rest).
        if self.cross_tokenizer:
            args.use_uld_loss = True
            args.use_extended_uld = loss_mode == "gold"
            args.uld_use_hybrid_loss = loss_mode == "gold"

        # Match OPSD / GOLDConfig (`disable_dropout` defaults True): a stochastic teacher
        # forward would inject noise into the distillation target. Applied to both peers;
        # no-op for dropout-0 models (e.g. Qwen), real for models that carry dropout.
        if args.disable_dropout:
            disable_dropout_in_model(model1)
            disable_dropout_in_model(model2)

        model_pair = CoModelPair(
            model1,
            model2,
            beta=args.beta,
            temperature=args.temperature,
            jsd_token_clip=getattr(args, "jsd_token_clip", None),
            use_ema_teacher=use_ema_teacher,
            ema_decay=ema_decay,
        )

        super().__init__(
            model=model_pair,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=tokenizer1,
            callbacks=callbacks,
        )

        # Build the per-direction ULD/GOLD loss now that the accelerator device
        # is known (ULDLoss places its vocabulary-mapping tensor on `device`).
        # dir1: model1 student ← model2 teacher; dir2 swaps the tokenizers.
        if self.cross_tokenizer:
            dev = self.accelerator.device
            self.model.uld_dir1 = ULDLoss(args, tokenizer1, tokenizer2, device=dev)
            self.model.uld_dir2 = ULDLoss(args, tokenizer2, tokenizer1, device=dev)

        # vLLM colocate: one engine per model in every process. Built while the
        # training models are still on CPU (DeepSpeed moves them to GPU only in
        # `train()`), so each engine profiles against a near-empty GPU.
        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if LLM is None:
                raise ImportError("use_vllm=True but vLLM is not installed (`pip install vllm`).")
            if args.vllm_mode != "colocate":
                raise ValueError("co-OPSD only supports vllm_mode='colocate'.")
            self._init_vllm()
            self.add_callback(CoVLLMSyncCallback(self))

        # EMA teacher: refresh both peers' EMA after every optimizer step.
        if self.use_ema_teacher:
            self.add_callback(EMATeacherUpdateCallback(self))
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[co-OPSD] EMA teacher ON (decay={ema_decay}): each peer scores "
                      f"with its EMA weights, not its live (co-trained) weights.")

    def _generate(self, model, tokenizer, prompt_ids, prompt_mask):
        """Sample one on-policy trajectory batch from `model` given its student prompts."""
        original_use_cache = model.config.use_cache
        model.config.use_cache = True
        gen_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            pad_token_id=tokenizer.pad_token_id,
        )
        try:
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=gen_config,
                use_cache=True,
            )
        finally:
            model.config.use_cache = original_use_cache
        return generated

    def _init_vllm(self):
        """Create one colocated vLLM engine per model (tensor-parallel size 1)."""
        os.environ["RANK"] = str(self.accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
        ensure_master_addr_port()

        common = dict(
            tensor_parallel_size=1,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_model_len=self.args.max_length,
            max_num_seqs=self.args.per_device_train_batch_size,
            distributed_executor_backend="external_launcher",
            seed=self.accelerator.process_index,
            enforce_eager=True,
        )
        self.vllm_engine1 = LLM(model=self.model.model1.config._name_or_path, **common)
        self.vllm_engine2 = LLM(model=self.model.model2.config._name_or_path, **common)
        self.accelerator.wait_for_everyone()

    def _generate_vllm(self, engine, tokenizer, prompt_ids):
        """Generate one trajectory batch with a colocated vLLM engine.

        Prompts are decoded to clean text (padding stripped), generated, then the
        completions are padded to `max_completion_length` and concatenated onto
        the re-tokenized left-padded prompts — matching the HF-generate path's
        `[left_pad][prompt][traj]` layout. Returns `(full_ids, prompt_len)`.
        """
        device = prompt_ids.device
        max_new = self.args.max_completion_length

        texts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)
        if tokenizer.pad_token:
            texts = [t.replace(tokenizer.pad_token, "") for t in texts]

        top_k = self.args.top_k if self.args.top_k and self.args.top_k > 0 else -1
        sampling_params = SamplingParams(
            n=1,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=top_k,
            max_tokens=max_new,
        )
        outputs = engine.generate(texts, sampling_params=sampling_params, use_tqdm=False)

        # Pad/truncate every completion to a uniform length.
        pad_id = tokenizer.pad_token_id
        comp_rows = []
        for o in outputs:
            c = list(o.outputs[0].token_ids)[:max_new]
            c = c + [pad_id] * (max_new - len(c))
            comp_rows.append(c)
        completion_ids = torch.tensor(comp_rows, device=device)

        # Re-tokenize the cleaned prompts (left padding => uniform prompt end).
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        prompt_enc = tokenizer(
            texts, padding="longest", truncation=True, max_length=self.args.max_length,
            return_tensors="pt", add_special_tokens=False,
        )
        tokenizer.padding_side = prev_side
        prompt_ids_clean = prompt_enc["input_ids"].to(device)

        full_ids = torch.cat([prompt_ids_clean, completion_ids], dim=1)
        return full_ids, prompt_ids_clean.shape[1]

    def _sync_vllm(self):
        """Load both models' current weights into their colocated vLLM engines.

        For PEFT (LoRA) models we merge the adapter into the base weights in
        place, strip the PEFT name decorations so the parameter names match what
        vLLM's loader expects (``model.layers.0.self_attn.q_proj.weight``; vLLM
        fuses q/k/v itself), push the merged weights, then unmerge. This keeps
        generation on-policy — without it vLLM would keep sampling from the
        step-0 base model. Mirrors ``OPSDTrainer._move_model_to_vllm``; the
        accelerate config here is DeepSpeed ZeRO-2, so parameters are unsharded
        and need no gathering before merge.
        """
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None

        pair = self.accelerator.unwrap_model(self.model)
        for model, engine in ((pair.model1, self.vllm_engine1), (pair.model2, self.vllm_engine2)):
            llm_model = engine.llm_engine.model_executor.driver_worker.model_runner.model
            is_peft = PeftModel is not None and isinstance(model, PeftModel)
            if is_peft:
                model.merge_adapter()
                for name, param in model.named_parameters():
                    # Recover the original parameter name and discard LoRA factors.
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if model.prefix in name:  # "lora_" parameters
                        continue
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")
                    llm_model.load_weights([(name, param.data)])
                model.unmerge_adapter()
            else:
                llm_model.load_weights([(name, param.data) for name, param in model.named_parameters()])
            engine.reset_prefix_cache()

    def _retokenize(self, traj_ids, src_tokenizer, dst_tokenizer):
        """Decode a trajectory from `src_tokenizer` and re-encode it with `dst_tokenizer`.

        Used on the cross-tokenizer path so the scoring model sees the trajectory
        in *its own* tokenization. The re-encoded batch is right-padded so the
        trajectory (answer) region stays contiguous after the teacher prompt.
        """
        texts = src_tokenizer.batch_decode(traj_ids, skip_special_tokens=True)
        prev_side = dst_tokenizer.padding_side
        dst_tokenizer.padding_side = "right"
        enc = dst_tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        dst_tokenizer.padding_side = prev_side
        return enc["input_ids"].to(traj_ids.device)

    def _build_direction(self, student_full, student_prompt_len,
                         teacher_prompt_ids, teacher_prompt_len, teacher_traj_ids,
                         student_pad_id, teacher_pad_id):
        """Assemble one direction's tensors for `CoModelPair`.

        `student_full` is `[left_pad][student_prompt][traj]` (produced by
        generate); the teacher sequence is `[teacher_prompt][teacher_traj]`,
        where `teacher_traj` equals the student trajectory for a same-tokenizer
        pair and the re-tokenized trajectory for a cross-tokenizer pair. The loss
        is computed only on the trajectory, so both label tensors mask the whole
        prompt region and any padding.
        """
        student_mask = (student_full != student_pad_id).long()
        student_labels = student_full.clone()
        student_labels[:, :student_prompt_len] = -100
        student_labels[student_full == student_pad_id] = -100

        teacher_full = torch.cat([teacher_prompt_ids, teacher_traj_ids], dim=1)
        teacher_mask = (teacher_full != teacher_pad_id).long()
        teacher_labels = teacher_full.clone()
        teacher_labels[:, :teacher_prompt_len] = -100
        teacher_labels[teacher_full == teacher_pad_id] = -100

        return {
            "student_ids": student_full,
            "student_mask": student_mask,
            "student_prompt_len": student_prompt_len,
            "student_labels": student_labels,
            "teacher_ids": teacher_full,
            "teacher_mask": teacher_mask,
            "teacher_prompt_len": teacher_prompt_len,
            "teacher_labels": teacher_labels,
        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        # The dataloader batch is still on CPU here (the base `training_step`
        # only moves it after this override); move it before generation.
        inputs = self._prepare_inputs(inputs)

        # === GENERATION PHASE (no gradients) ===
        if self.use_vllm:
            m1_gen, m1_prompt_len = self._generate_vllm(
                self.vllm_engine1, self.tokenizer1, inputs["m1_student_input_ids"])
            m2_gen, m2_prompt_len = self._generate_vllm(
                self.vllm_engine2, self.tokenizer2, inputs["m2_student_input_ids"])
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                m1, m2 = unwrapped.model1, unwrapped.model2
                m1_gen = self._generate(
                    m1, self.tokenizer1,
                    inputs["m1_student_input_ids"], inputs["m1_student_attention_mask"])
                m1_prompt_len = inputs["m1_student_prompt_len"]
                m2_gen = self._generate(
                    m2, self.tokenizer2,
                    inputs["m2_student_input_ids"], inputs["m2_student_attention_mask"])
                m2_prompt_len = inputs["m2_student_prompt_len"]
        traj1 = m1_gen[:, m1_prompt_len:]
        traj2 = m2_gen[:, m2_prompt_len:]

        pad1 = self.tokenizer1.pad_token_id
        pad2 = self.tokenizer2.pad_token_id

        # The teacher scores the trajectory in its own tokenization. Same
        # tokenizer → reuse the student tokens; cross tokenizer → re-encode.
        if self.cross_tokenizer:
            teacher_traj_for_1 = self._retokenize(traj1, self.tokenizer1, self.tokenizer2)
            teacher_traj_for_2 = self._retokenize(traj2, self.tokenizer2, self.tokenizer1)
        else:
            teacher_traj_for_1 = traj1
            teacher_traj_for_2 = traj2

        # Direction 1: model1 student (traj1), model2 teacher.
        dir1 = self._build_direction(
            student_full=m1_gen,
            student_prompt_len=m1_prompt_len,
            teacher_prompt_ids=inputs["m2_teacher_input_ids"],
            teacher_prompt_len=inputs["m2_teacher_prompt_len"],
            teacher_traj_ids=teacher_traj_for_1,
            student_pad_id=pad1,
            teacher_pad_id=pad2,
        )
        # Direction 2: model2 student (traj2), model1 teacher.
        dir2 = self._build_direction(
            student_full=m2_gen,
            student_prompt_len=m2_prompt_len,
            teacher_prompt_ids=inputs["m1_teacher_input_ids"],
            teacher_prompt_len=inputs["m1_teacher_prompt_len"],
            teacher_traj_ids=teacher_traj_for_2,
            student_pad_id=pad2,
            teacher_pad_id=pad1,
        )

        step_inputs = {f"dir1_{k}": v for k, v in dir1.items()}
        step_inputs.update({f"dir2_{k}": v for k, v in dir2.items()})

        if random.random() < 0.02:
            sample = self.tokenizer1.decode(traj1[0], skip_special_tokens=True)
            print(f"\n[co-OPSD step {self.state.global_step}] model1 traj sample:\n{sample[:600]}\n")

        return super().training_step(model, step_inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        # Track each direction's loss so `log` can surface model1 vs model2.
        self._loss1_log.append(float(outputs["loss1"]))
        self._loss2_log.append(float(outputs["loss2"]))
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        # Trainer logs the total `loss` (= loss1 + loss2); also surface each
        # direction's mean since they belong to two separately-updated models.
        if self._loss1_log:
            logs["loss1"] = sum(self._loss1_log) / len(self._loss1_log)
            logs["loss2"] = sum(self._loss2_log) / len(self._loss2_log)
            self._loss1_log.clear()
            self._loss2_log.clear()
        super().log(logs, start_time)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the two models into separate subdirectories."""
        output_dir = output_dir or self.args.output_dir
        pair = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            pair.model1.save_pretrained(f"{output_dir}/model1")
            self.tokenizer1.save_pretrained(f"{output_dir}/model1")
            pair.model2.save_pretrained(f"{output_dir}/model2")
            self.tokenizer2.save_pretrained(f"{output_dir}/model2")
