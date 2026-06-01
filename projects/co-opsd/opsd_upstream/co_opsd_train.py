"""co-OPSD training entrypoint: two-model on-policy co-distillation.

Launches a `CoOPSDTrainer` over a model pair. The two models may share an
initialization, share a family, or come from different families — the data
layer and the ULD/GOLD loss path keep the setup tokenizer-agnostic.

The two models' data streams are configured independently through the
`--model{1,2}_dataset / _split / _shuffle_seed` arguments; see `co_opsd_data.py`
for how the four data regimes (same / shuffled / different / subset) map onto
those three fields.
"""

import copy
import os
from dataclasses import dataclass, field

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import TrlParser
from trl.experimental.gold import GOLDConfig

from co_opsd_data import ModelDataSpec, CoSelfDistillationDataCollator, build_paired_dataset
from co_opsd_trainer import CoOPSDTrainer

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CoOPSDScriptArguments:
    """co-OPSD-specific arguments (model pair, per-model data streams, loss knobs)."""

    model1_name_or_path: str = field(metadata={"help": "Path/name of model1."})
    model2_name_or_path: str = field(metadata={"help": "Path/name of model2."})

    # model1 data stream
    model1_dataset: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd", metadata={"help": "model1 dataset."}
    )
    model1_split: str = field(default="train", metadata={"help": "model1 split (slice for subsets)."})
    model1_shuffle_seed: int = field(default=42, metadata={"help": "model1 shuffle seed."})
    model1_problem_column: str = field(default="problem", metadata={"help": "model1 problem column."})
    model1_solution_column: str = field(default="solution", metadata={"help": "model1 solution column."})

    # model2 data stream
    model2_dataset: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd", metadata={"help": "model2 dataset."}
    )
    model2_split: str = field(default="train", metadata={"help": "model2 split (slice for subsets)."})
    model2_shuffle_seed: int = field(default=42, metadata={"help": "model2 shuffle seed."})
    model2_problem_column: str = field(default="problem", metadata={"help": "model2 problem column."})
    model2_solution_column: str = field(default="solution", metadata={"help": "model2 solution column."})

    teacher_sees_gt_answer: bool = field(
        default=True,
        metadata={
            "help": "Whether the scoring (teacher) prompt embeds the ground-truth solution "
            "(OPSD's privileged teacher). If False, the only signal is the two models' diversity."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "Per-token JSD clip; 0 disables clipping."},
    )
    distill_loss_type: str = field(
        default="auto",
        metadata={
            "help": "Distillation loss: 'jsd' = exact generalized JSD; 'uld' = original "
            "Universal Logit Distillation (positional truncation + sorted-probability "
            "distance); 'gold' = GOLD (token-merging alignment + hybrid JSD/ULD loss); "
            "'auto' = 'jsd' for a same-tokenizer pair, 'gold' otherwise. A same-tokenizer "
            "pair always uses 'jsd'. Use 'uld' vs 'gold' to ablate the two cross-tokenizer losses."
        },
    )
    run_config: str = field(default=None, metadata={"help": "Run name for output dir and WandB."})
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"help": "Attention implementation for both models."}
    )

    # ---- LoRA support (both models get an independent LoRA adapter) ----
    use_peft: bool = field(
        default=False,
        metadata={
            "help": "Wrap both models with LoRA adapters before training. Hparams aligned with "
            "OPSD's --use_peft path (run_opsd_1b.sh). Reduces per-ckpt size 40x and is much "
            "less prone to gradient collapse than full FT."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA rank (matches OPSD upstream default)."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha (matches OPSD upstream default)."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "LoRA target module names (Qwen/Llama transformer block conventions)."},
    )


def _load_model(path, attn_implementation, gradient_checkpointing):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        use_cache=not gradient_checkpointing,
    )


def main():
    parser = TrlParser((CoOPSDScriptArguments, GOLDConfig))
    script_args, training_args = parser.parse_args_and_config()

    # jsd_token_clip is read off training_args by CoModelPair.
    training_args.jsd_token_clip = script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None

    if script_args.run_config and not training_args.output_dir.endswith(script_args.run_config):
        training_args.output_dir = os.path.join(training_args.output_dir, script_args.run_config)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=script_args.run_config,
            config={
                "model1": script_args.model1_name_or_path,
                "model2": script_args.model2_name_or_path,
                "teacher_sees_gt_answer": script_args.teacher_sees_gt_answer,
                "beta": training_args.beta,
                "temperature": training_args.temperature,
                "jsd_token_clip": training_args.jsd_token_clip,
            },
        )

    # === Models & tokenizers ===
    tokenizer1 = AutoTokenizer.from_pretrained(script_args.model1_name_or_path, padding_side="left")
    tokenizer2 = AutoTokenizer.from_pretrained(script_args.model2_name_or_path, padding_side="left")
    for tok in (tokenizer1, tokenizer2):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    model1 = _load_model(
        script_args.model1_name_or_path, script_args.attn_implementation, training_args.gradient_checkpointing
    )
    model2 = _load_model(
        script_args.model2_name_or_path, script_args.attn_implementation, training_args.gradient_checkpointing
    )

    # Optional: wrap both models with an independent LoRA adapter. Mirrors OPSD's
    # --use_peft path. Saves only adapter weights (~230 MB per model vs ~6 GB
    # full FT), and the small per-step update keeps cross-tokenizer GOLD loss
    # much more stable than full-FT co-OPSD's exploding-gradient runs.
    if script_args.use_peft:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Each get_peft_model mutates its config's base_model_name_or_path, so
        # give the second model its own copy to avoid a stale (model1) base name.
        model1 = get_peft_model(model1, lora_config)
        model2 = get_peft_model(model2, copy.deepcopy(lora_config))
        # LoRA + gradient_checkpointing fix: embeddings are frozen (no LoRA on
        # them), so without this, gradient checkpointing's saved tensors lose
        # their grad_fn and DeepSpeed's `assert maybe_loss_for_backward(loss)`
        # fires with "loss must be a scalar tensor". `enable_input_require_grads`
        # makes the embedding output require_grad even though embedding weights
        # are frozen, restoring the computation graph through checkpoint blocks.
        if training_args.gradient_checkpointing:
            model1.enable_input_require_grads()
            model2.enable_input_require_grads()
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[co-OPSD] LoRA enabled: r={script_args.lora_r}, alpha={script_args.lora_alpha}, "
                  f"targets={script_args.lora_target_modules}")
            model1.print_trainable_parameters()

    # === Data ===
    spec1 = ModelDataSpec(
        dataset=script_args.model1_dataset,
        split=script_args.model1_split,
        shuffle_seed=script_args.model1_shuffle_seed,
        problem_column=script_args.model1_problem_column,
        solution_column=script_args.model1_solution_column,
    )
    spec2 = ModelDataSpec(
        dataset=script_args.model2_dataset,
        split=script_args.model2_split,
        shuffle_seed=script_args.model2_shuffle_seed,
        problem_column=script_args.model2_problem_column,
        solution_column=script_args.model2_solution_column,
    )
    train_dataset = build_paired_dataset(spec1, spec2)
    data_collator = CoSelfDistillationDataCollator(
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        max_length=training_args.max_length,
        teacher_sees_gt_answer=script_args.teacher_sees_gt_answer,
    )

    # === Train ===
    trainer = CoOPSDTrainer(
        model1=model1,
        model2=model2,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        teacher_sees_gt_answer=script_args.teacher_sees_gt_answer,
        distill_loss_type=script_args.distill_loss_type,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
