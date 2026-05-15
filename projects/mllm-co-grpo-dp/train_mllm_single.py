"""Single-model VLM GRPO baseline (Phase 3 sanity, no cross-supervision).

Trains one VLM with standard GRPO using the **dataset's ground-truth
solution** as the reward target — no peer rendezvous, no pseudo-labels,
no `<group>` arg. Use this to verify the VLM training loop is healthy
and that inline eval produces reasonable numbers before flipping on
cross-supervision (Phase 4 via `train_mllm_co_grpo_dp.py`).

Shares with `train_mllm_co_grpo_dp.py`:
  - `AutoProcessor` (VLM)
  - `dataset.load_dataset(--train_dataset)`  (R1-V prompt + image)
  - `reward_correctness` (math_verify grader, `<answer>` extractor)

Does NOT share:
  - Rendezvous (not needed — no peer)
  - `--group` / `--peer_model_name_or_path` / `--rendezvous_dir`
  - Dual-seed bump (no peer to diverge from)
  - `CoGRPOdpTrainer` — uses vanilla `GRPOTrainer`
"""

import os
from dataclasses import dataclass, field

import wandb
from transformers import AutoProcessor

from co_label_utils import extract_boxed_answer, grade_answer
from dataset import CLEVR_COUNTING_DATASET, GEOQA_DATASET, load_dataset

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class MllmSingleScriptArguments(ScriptArguments):
    """Script arguments for Phase 3 single-model VLM baseline."""

    run_config: str = field(default=None, metadata={"help": "Run name prefix."})
    wandb_entity: str = field(default=None, metadata={"help": "WandB entity."})
    wandb_project: str = field(default="mllm-co-grpo-dp", metadata={"help": "WandB project name."})
    train_dataset: str = field(
        default=CLEVR_COUNTING_DATASET,
        metadata={
            "help": "Dataset to use for training.",
            "choices": [CLEVR_COUNTING_DATASET, GEOQA_DATASET],
        },
    )


def _get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def reward_correctness(completions, solution, **kwargs):
    """1.0 iff completion's `<answer>` content is math-equivalent to GT."""
    rewards = []
    for completion, ground_truth in zip(completions, solution):
        pred_answer = extract_boxed_answer(_get_text(completion))
        if pred_answer is not None and grade_answer(pred_answer, ground_truth):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


if __name__ == "__main__":
    parser = TrlParser((MllmSingleScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    model_short = model_args.model_name_or_path.split("/")[-1]
    full_wandb_run_name = (
        script_args.run_config
        or f"MllmSingle_{model_short}_lr{lr_str}_bs{effective_batch_size}"
    )

    print(f"\n{'='*80}")
    print(f"MLLM-SINGLE (Phase 3 baseline) CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model        : {model_args.model_name_or_path}")
    print(f"Dataset      : {script_args.train_dataset}")
    print(f"WandB run    : {full_wandb_run_name}")
    print(f"Output dir   : {training_args.output_dir}")
    print(f"World size   : {num_processes}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model": model_args.model_name_or_path,
                "train_dataset": script_args.train_dataset,
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
                "steps_per_generation": training_args.steps_per_generation,
                "vllm_importance_sampling_correction": training_args.vllm_importance_sampling_correction,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "warmup_ratio": training_args.warmup_ratio,
                "eval_steps": training_args.eval_steps,
                "num_generations_eval": training_args.num_generations_eval,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "vllm_gpu_memory_utilization": training_args.vllm_gpu_memory_utilization,
                "seed": training_args.seed,
            },
        )

    ################
    # Model & Processor
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

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_dataset, eval_dataset = load_dataset(script_args.train_dataset)

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
