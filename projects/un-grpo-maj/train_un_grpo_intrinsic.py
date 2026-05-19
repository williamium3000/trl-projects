"""Entry point for un-grpo-maj training with intrinsic single-model rewards.

Two reward types selected at launch via `--reward_type`:
  - `entropy` (RENT, Prabhudesai et al. 2025, arXiv 2505.22660):
        r = -mean_t H(p_t)  — maximize negative entropy = increase confidence
  - `self_certainty` (Intuitor, Zhao et al. 2025, arXiv 2505.19590):
        r = mean_t KL(U || p_t)  — maximize mode-seeking divergence from uniform

Both are fully intrinsic (no peer, no self-labeling, no ground truth). The
trainer (`IntrinsicRewardTrainer`) computes the per-token quantity via a chunked
no-grad forward and stashes a per-rollout scalar into `inputs[i]["intrinsic_reward"]`,
which the closure-bound reward function surfaces to TRL.

Eval mode skips the chunked forward; the reward closure detects missing
`intrinsic_reward` kwargs and falls back to binary equality against the dataset's
`solution`, matching baseline `reward_correctness`.

See `projects/un-grpo-maj/intrinsic_rewards.py` and `intrinsic_trainer.py`.
"""

import os
import sys
from dataclasses import dataclass, field

import wandb
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intrinsic_rewards import make_reward_entropy, make_reward_self_certainty
from intrinsic_trainer import IntrinsicRewardTrainer
from dataset import DAPO_DATASET, MATH_LEVEL12345_DATASET, MATH_LEVEL345_DATASET, OPSD_DATASET, load_dataset

from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class IntrinsicScriptArguments(ScriptArguments):
    """Script arguments for un-grpo-maj intrinsic-reward variant."""

    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Used for both output_dir suffix "
            "and WandB run name."
        },
    )
    wandb_entity: str = field(default=None, metadata={"help": "WandB entity."})
    wandb_project: str = field(
        default="un-grpo-maj",
        metadata={"help": "WandB project name."},
    )
    train_dataset: str = field(
        default=OPSD_DATASET,
        metadata={
            "help": "Dataset to use for GRPO training.",
            "choices": [OPSD_DATASET, DAPO_DATASET, MATH_LEVEL345_DATASET, MATH_LEVEL12345_DATASET],
        },
    )
    reward_type: str = field(
        default="entropy",
        metadata={
            "help": "Which intrinsic reward. 'entropy' (RENT, arXiv 2505.22660) "
            "uses -mean_t H(p_t). 'self_certainty' (Intuitor, arXiv 2505.19590 Eq.2) "
            "uses mean_t KL(U||p_t). The two are NOT affine equivalent — paper main "
            "table reports them separately.",
            "choices": ["entropy", "self_certainty"],
        },
    )
    intrinsic_chunk_size: int = field(
        default=4,
        metadata={
            "help": "Number of rollouts per chunked forward batch when computing "
            "per-token entropy / KL(U||p). Lower this if OOM (each chunk's peak "
            "memory scales with chunk_size * T * V * 4 bytes fp32)."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((IntrinsicScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name & Output Directory
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_name = (
            f"UnGRPOIntrinsic_{script_args.reward_type}_{model_name}_"
            f"lr{lr_str}_bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}"
        )

    print(f"\n{'='*80}")
    print(f"UN-GRPO-MAJ INTRINSIC ({script_args.reward_type.upper()}) RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"Reward type     : {script_args.reward_type}")
    print(f"Chunk size      : {script_args.intrinsic_chunk_size}")
    print(f"WandB run       : {full_wandb_run_name}")
    print(f"Output dir      : {training_args.output_dir}")
    print(f"Num generations : {training_args.num_generations}")
    print(f"Temperature     : {training_args.temperature}")
    print(f"Max completion  : {training_args.max_completion_length}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "loss_type": training_args.loss_type,
                "scale_rewards": training_args.scale_rewards,
                "reward_type": script_args.reward_type,
                "intrinsic_chunk_size": script_args.intrinsic_chunk_size,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Attention impl: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    train_dataset, eval_dataset = load_dataset(script_args.train_dataset)

    ################
    # Reward function dispatch
    ################
    if script_args.reward_type == "entropy":
        reward_fn = make_reward_entropy()
    elif script_args.reward_type == "self_certainty":
        reward_fn = make_reward_self_certainty()
    else:
        raise ValueError(f"Unknown --reward_type {script_args.reward_type!r}")

    ################
    # Training
    ################
    trainer = IntrinsicRewardTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        intrinsic_reward_type=script_args.reward_type,
        intrinsic_chunk_size=script_args.intrinsic_chunk_size,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
