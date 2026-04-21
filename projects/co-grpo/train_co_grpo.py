"""Entry point for co-learning unsupervised GRPO with two models."""

import os
import wandb

from transformers import AutoTokenizer
from dataset import DAPO_DATASET, MATH_LEVEL12345_DATASET, MATH_LEVEL345_DATASET, OPSD_DATASET, load_dataset
from co_grpo_trainer import CoGRPOTrainer
from co_label_utils import extract_boxed_answer, normalize_answer

from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field


@dataclass
class CoGRPOScriptArguments(ScriptArguments):
    """Script arguments for co-learning GRPO."""

    run_config: str = field(
        default=None,
        metadata={"help": "Run name for this experiment."},
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "WandB entity."},
    )
    wandb_project: str = field(
        default="co-grpo",
        metadata={"help": "WandB project name."},
    )
    train_dataset: str = field(
        default=OPSD_DATASET,
        metadata={
            "help": "Dataset to use for training.",
            "choices": [OPSD_DATASET, DAPO_DATASET, MATH_LEVEL345_DATASET, MATH_LEVEL12345_DATASET],
        },
    )
    self_consistency_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Minimum top-answer frequency for a pseudo-label to be accepted. "
            "0.0 accepts the plurality winner; 0.5 requires a strict majority."
        },
    )
    log_oracle_accuracy: bool = field(
        default=True,
        metadata={
            "help": "Log how often pseudo-labels match real ground truth (diagnostic only)."
        },
    )

    # ---- Model B configuration ----
    model_name_or_path_b: str = field(
        default=None,
        metadata={"help": "Path or name of the second model."},
    )
    use_peft_b: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT/LoRA for model B."},
    )
    lora_r_b: int = field(
        default=16,
        metadata={"help": "LoRA rank for model B."},
    )
    lora_alpha_b: int = field(
        default=32,
        metadata={"help": "LoRA alpha for model B."},
    )
    lora_target_modules_b: list[str] = field(
        default=None,
        metadata={"help": "LoRA target modules for model B. Defaults to same as model A."},
    )
    learning_rate_b: float = field(
        default=None,
        metadata={"help": "Learning rate for model B. Defaults to model A's learning rate."},
    )
    gpu_memory_utilization: float = field(
        default=0.35,
        metadata={"help": "vLLM GPU memory utilization for model A (default 0.35 since two engines share GPUs)."},
    )
    gpu_memory_utilization_b: float = field(
        default=0.35,
        metadata={"help": "vLLM GPU memory utilization for model B."},
    )


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def reward_correctness(completions, solution, **kwargs):
    """Reward function: 1.0 if \\boxed{} answer matches solution, else 0.0."""
    rewards = []
    for completion, ground_truth in zip(completions, solution):
        pred_answer = extract_boxed_answer(_get_text(completion))
        pred_normalized = normalize_answer(pred_answer)
        gt_normalized = normalize_answer(ground_truth)
        if pred_normalized is not None and pred_normalized == gt_normalized:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def _build_peft_config_b(script_args, model_args):
    """Build a PEFT config for model B from script args."""
    if not script_args.use_peft_b:
        return None

    from peft import LoraConfig

    # Default target modules: use model A's if not specified
    target_modules = script_args.lora_target_modules_b
    if target_modules is None and model_args.use_peft:
        target_modules = model_args.lora_target_modules

    return LoraConfig(
        r=script_args.lora_r_b,
        lora_alpha=script_args.lora_alpha_b,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )


if __name__ == "__main__":
    parser = TrlParser((CoGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.model_name_or_path_b is None:
        raise ValueError("--model_name_or_path_b is required for co-learning GRPO.")

    ################
    # WandB
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    model_a_short = model_args.model_name_or_path.split("/")[-1]
    model_b_short = script_args.model_name_or_path_b.split("/")[-1]

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        full_wandb_run_name = (
            f"CoGRPO_{model_a_short}_x_{model_b_short}_"
            f"lr{lr_str}_bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}_"
            f"sct{script_args.self_consistency_threshold}"
        )

    print(f"\n{'='*80}")
    print(f"CO-LEARNING GRPO CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model A: {model_args.model_name_or_path}")
    print(f"Model B: {script_args.model_name_or_path_b}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Self-Consistency Threshold: {script_args.self_consistency_threshold}")
    print(f"GPU Memory Utilization: A={script_args.gpu_memory_utilization}, B={script_args.gpu_memory_utilization_b}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_a": model_args.model_name_or_path,
                "model_b": script_args.model_name_or_path_b,
                "learning_rate": training_args.learning_rate,
                "learning_rate_b": script_args.learning_rate_b or training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft_a": model_args.use_peft,
                "use_peft_b": script_args.use_peft_b,
                "lora_r_a": model_args.lora_r if model_args.use_peft else None,
                "lora_r_b": script_args.lora_r_b if script_args.use_peft_b else None,
                "self_consistency_threshold": script_args.self_consistency_threshold,
                "gpu_memory_utilization_a": script_args.gpu_memory_utilization,
                "gpu_memory_utilization_b": script_args.gpu_memory_utilization_b,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

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

    # Tokenizer A
    tokenizer_a = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token

    # Tokenizer B
    tokenizer_b = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path_b,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token

    ################
    # Dataset
    ################
    train_dataset, eval_dataset = load_dataset(script_args.train_dataset)

    ################
    # PEFT configs
    ################
    peft_config_a = get_peft_config(model_args)
    peft_config_b = _build_peft_config_b(script_args, model_args)

    ################
    # Training
    ################
    trainer = CoGRPOTrainer(
        model=model_args.model_name_or_path,
        model_b=script_args.model_name_or_path_b,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer_a,
        processing_class_b=tokenizer_b,
        peft_config=peft_config_a,
        peft_config_b=peft_config_b,
        self_consistency_threshold=script_args.self_consistency_threshold,
        log_oracle_accuracy=script_args.log_oracle_accuracy,
        learning_rate_b=script_args.learning_rate_b,
        gpu_memory_utilization_b=script_args.gpu_memory_utilization_b,
    )

    trainer.train()

    # Save both models
    trainer.save_model(training_args.output_dir)
