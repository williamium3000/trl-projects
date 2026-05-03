"""Entry point for un-grpo-maj training with 4-regime confidence-gated reward.

Same launch shape as `train_un_grpo.py` (single accelerate world, no peer
coordination), but the reward function consumes the per-prompt list of self
rollout answers and applies a 4-regime confidence-gated reward instead of
binary equality against the self-majority pseudo-label. See
`self_label_utils.compute_4regime_reward` for the reward formula and regime
semantics.

The eval path is identical to baseline: `inputs[i]["self_answers"]` is not
written in eval mode, so `reward_4regime` falls back to binary equality
against the dataset's ground-truth `solution`.
"""

import os
import sys
from dataclasses import dataclass, field

import wandb
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_label_utils import (
    _extract_and_normalize,
    compute_4regime_reward,
    extract_boxed_answer,
    grade_answer,
)
from dataset import DAPO_DATASET, MATH_LEVEL12345_DATASET, MATH_LEVEL345_DATASET, OPSD_DATASET, load_dataset
from self_label_4regime_trainer import SelfLabel4RegimeTrainer

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
class CustomScriptArguments(ScriptArguments):
    """Script arguments for un-grpo-maj 4-regime variant."""

    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name."
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
    self_consistency_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Minimum top-answer frequency among parseable rollouts for a prompt's "
            "self_answers payload to be retained. Below this, the payload is encoded as "
            "[None]*G and the 4-regime reward returns 0 (no_valid_peer regime)."
        },
    )
    log_oracle_accuracy: bool = field(
        default=True,
        metadata={
            "help": "If True, log how often the self-majority matches the real solution. "
            "Diagnostic only — does not affect training."
        },
    )
    tau_high: float = field(
        default=0.625,
        metadata={
            "help": "High-confidence threshold for 4-regime reward. With num_generations=8, "
            "0.625 = 5/8."
        },
    )
    tau_mid: float = field(
        default=0.25,
        metadata={
            "help": "Strong-runner-up threshold for 4-regime reward. With num_generations=8, "
            "0.25 = 2/8."
        },
    )
    lambda_4regime: float = field(
        default=0.5,
        metadata={"help": "Negative penalty magnitude for weak / unseen rollouts."},
    )


def _get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def make_reward_4regime(tau_high, tau_mid, lambda_):
    """Return a closure-bound reward function with 4-regime hyperparameters baked in.

    Dispatches on `self_answers`:
      - train mode: each completion is scored via `compute_4regime_reward`
        against its prompt's self answer list. Per-batch regime occupancy is
        logged via `log_metric` when available.
      - eval mode (self_answers is None — trainer didn't write it): each
        completion is scored 1.0 / 0.0 against `solution[i]` via sympy
        `grade_answer`. Identical to baseline `reward_correctness`.
    """

    def reward_4regime(completions, self_answers=None, solution=None, log_metric=None, **kwargs):
        rewards = []

        if self_answers is None:
            # Eval mode: binary against ground truth.
            for completion, gt in zip(completions, solution):
                pred_raw = extract_boxed_answer(_get_text(completion))
                rewards.append(1.0 if pred_raw is not None and grade_answer(pred_raw, gt) else 0.0)
            return rewards

        # Train mode: 4-regime against self answer distribution.
        regime_counts = {
            "confident_positive": 0,
            "uncertain_majority": 0,
            "strong_runner_up": 0,
            "weak": 0,
            "unseen": 0,
            "no_valid_peer": 0,
            "tied_top": 0,
        }
        for completion, peer in zip(completions, self_answers):
            pred_canonical = _extract_and_normalize(completion)
            r, regime = compute_4regime_reward(pred_canonical, peer, tau_high, tau_mid, lambda_)
            rewards.append(r)
            regime_counts[regime] += 1

        if log_metric is not None and len(completions) > 0:
            total = len(completions)
            for regime, cnt in regime_counts.items():
                log_metric(f"4regime/pct_{regime}", cnt / total)
            log_metric("4regime/avg_reward", sum(rewards) / total)

        return rewards

    return reward_4regime


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name & Output Directory
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_name = (
            f"UnGRPO4Regime_{model_name}_"
            f"lr{lr_str}_bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}_"
            f"th{script_args.tau_high}_tm{script_args.tau_mid}_lam{script_args.lambda_4regime}"
        )

    print(f"\n{'='*80}")
    print(f"UN-GRPO-MAJ 4-REGIME RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Num Generations: {training_args.num_generations}")
    print(f"Temperature: {training_args.temperature}")
    print(f"Max Completion Length: {training_args.max_completion_length}")
    print(f"Self-Consistency Threshold: {script_args.self_consistency_threshold}")
    print(f"tau_high: {script_args.tau_high}")
    print(f"tau_mid : {script_args.tau_mid}")
    print(f"lambda  : {script_args.lambda_4regime}")
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
                "self_consistency_threshold": script_args.self_consistency_threshold,
                "tau_high": script_args.tau_high,
                "tau_mid": script_args.tau_mid,
                "lambda_4regime": script_args.lambda_4regime,
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
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
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
    # Training
    ################
    trainer = SelfLabel4RegimeTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=make_reward_4regime(
            tau_high=script_args.tau_high,
            tau_mid=script_args.tau_mid,
            lambda_=script_args.lambda_4regime,
        ),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        self_consistency_threshold=script_args.self_consistency_threshold,
        log_oracle_accuracy=script_args.log_oracle_accuracy,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
