"""Entry point for one half of a cross-supervised GRPO data-parallel run with
4-regime confidence-gated reward.

Same launch shape as `train_co_grpo_dp.py` (one accelerate world per group, two
launches per experiment, file-rendezvous coordination), but the reward function
consumes a per-prompt peer-answer distribution instead of a single peer
majority. See `co_label_utils.compute_4regime_reward` for the exact reward
formula and regime semantics.

The eval path is identical to baseline: `inputs[i]["peer_answers"]` is not
written in eval mode, so `reward_4regime` falls back to binary equality against
the dataset's ground-truth `solution`.
"""

import os
from dataclasses import dataclass, field

import wandb
from transformers import AutoTokenizer

from co_grpo_dp_4regime_trainer import CoGRPOdp4RegimeTrainer
from co_label_utils import (
    _extract_and_normalize,
    compute_4regime_reward,
    extract_boxed_answer,
    grade_answer,
)
from dataset import DAPO_DATASET, MATH_LEVEL12345_DATASET, MATH_LEVEL345_DATASET, OPSD_DATASET, load_dataset
from rendezvous import Rendezvous

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
class CoGRPOdp4RegimeScriptArguments(ScriptArguments):
    """Script arguments for co-grpo-dp 4-regime variant."""

    group: str = field(
        default=None,
        metadata={"help": "'A' or 'B' — which half of the cross-supervision run this launch is."},
    )
    rendezvous_dir: str = field(
        default=None,
        metadata={"help": "Directory shared between groups for pseudo-label exchange."},
    )
    peer_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Peer group's model id (for logging only; peer is launched separately)."},
    )
    run_config: str = field(
        default=None,
        metadata={"help": "Run name prefix for this experiment."},
    )
    wandb_entity: str = field(default=None, metadata={"help": "WandB entity."})
    wandb_project: str = field(default="co-grpo-dp", metadata={"help": "WandB project name."})
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
            "help": "Minimum top-answer frequency among parseable peer rollouts for that prompt's "
            "peer payload to be retained. Below this, the peer payload is encoded as [None]*G and "
            "the 4-regime reward returns 0 (no_valid_peer regime)."
        },
    )
    log_oracle_accuracy: bool = field(
        default=True,
        metadata={"help": "Log how often pseudo-labels match real ground truth (diagnostic only)."},
    )
    tau_high: float = field(
        default=0.625,
        metadata={
            "help": "High-confidence threshold for 4-regime reward. With num_generations=8, "
            "0.625 = 5/8 (majority needs >= 5 votes to give +1). See compute_4regime_reward."
        },
    )
    tau_mid: float = field(
        default=0.25,
        metadata={
            "help": "Strong-runner-up threshold for 4-regime reward. With num_generations=8, "
            "0.25 = 2/8 (an answer needs to appear >= 2 times to count as strong runner-up "
            "and receive 0 reward instead of -lambda)."
        },
    )
    lambda_4regime: float = field(
        default=0.5,
        metadata={"help": "Negative penalty magnitude for weak / unseen rollouts in 4-regime reward."},
    )


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def make_reward_4regime(tau_high, tau_mid, lambda_):
    """Return a closure-bound reward function with 4-regime hyperparameters baked in.

    The returned function dispatches on `peer_answers`:
      - train mode (peer_answers is a list of length len(completions)): each
        completion is scored via `compute_4regime_reward` against its peer's
        per-prompt answer list. Per-batch regime occupancy is logged via
        `log_metric` when available.
      - eval mode (peer_answers is None — trainer didn't write it): each
        completion is scored 1.0 / 0.0 against `solution[i]` via sympy
        `grade_answer`. Identical to baseline `reward_correctness`.

    Args:
        tau_high (`float`):
            High-confidence threshold (e.g. 0.625 = 5/8 for N=8).
        tau_mid (`float`):
            Strong-runner-up threshold (e.g. 0.25 = 2/8 for N=8).
        lambda_ (`float`):
            Negative penalty magnitude (e.g. 0.5).

    Returns:
        `callable`: reward function with signature
        `(completions, peer_answers=None, solution=None, log_metric=None, **kwargs) -> list[float]`.
    """

    def reward_4regime(completions, peer_answers=None, solution=None, log_metric=None, **kwargs):
        rewards = []

        if peer_answers is None:
            # Eval mode: binary against ground truth (matches baseline reward_correctness).
            for completion, gt in zip(completions, solution):
                pred_raw = extract_boxed_answer(_get_text(completion))
                rewards.append(1.0 if pred_raw is not None and grade_answer(pred_raw, gt) else 0.0)
            return rewards

        # Train mode: 4-regime against peer answer distribution.
        regime_counts = {
            "confident_positive": 0,
            "uncertain_majority": 0,
            "strong_runner_up": 0,
            "weak": 0,
            "unseen": 0,
            "no_valid_peer": 0,
            "tied_top": 0,
        }
        for completion, peer in zip(completions, peer_answers):
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
    parser = TrlParser((CoGRPOdp4RegimeScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.group not in ("A", "B"):
        raise ValueError(f"--group must be 'A' or 'B', got {script_args.group!r}")
    if script_args.rendezvous_dir is None:
        raise ValueError("--rendezvous_dir is required for co-grpo-dp.")

    ################
    # WandB
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    model_short = model_args.model_name_or_path.split("/")[-1]
    peer_short = (
        script_args.peer_model_name_or_path.split("/")[-1]
        if script_args.peer_model_name_or_path
        else "unknown"
    )

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_group{script_args.group}_lr{lr_str}_bs{effective_batch_size}"
    else:
        full_wandb_run_name = (
            f"CoGRPOdp4Regime_{model_short}_x_{peer_short}_group{script_args.group}_"
            f"lr{lr_str}_bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}_"
            f"th{script_args.tau_high}_tm{script_args.tau_mid}_lam{script_args.lambda_4regime}"
        )

    print(f"\n{'='*80}")
    print(f"CO-GRPO-DP 4-REGIME (group {script_args.group}) CONFIGURATION")
    print(f"{'='*80}")
    print(f"This model   : {model_args.model_name_or_path}")
    print(f"Peer model   : {script_args.peer_model_name_or_path}")
    print(f"Rendezvous   : {script_args.rendezvous_dir}")
    print(f"WandB run    : {full_wandb_run_name}")
    print(f"Output dir   : {training_args.output_dir}")
    print(f"SCT          : {script_args.self_consistency_threshold}")
    print(f"tau_high     : {script_args.tau_high}")
    print(f"tau_mid      : {script_args.tau_mid}")
    print(f"lambda       : {script_args.lambda_4regime}")
    print(f"World size   : {num_processes}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "group": script_args.group,
                "model": model_args.model_name_or_path,
                "peer_model": script_args.peer_model_name_or_path,
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
                "self_consistency_threshold": script_args.self_consistency_threshold,
                "tau_high": script_args.tau_high,
                "tau_mid": script_args.tau_mid,
                "lambda_4regime": script_args.lambda_4regime,
                "vllm_gpu_memory_utilization": training_args.vllm_gpu_memory_utilization,
                "seed": training_args.seed,
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset — two groups use the same seed/world_size so RepeatSampler
    # yields identical index sequences, ensuring both groups train on the
    # same prompts at every generation step (required for cross-labeling).
    ################
    train_dataset, eval_dataset = load_dataset(script_args.train_dataset)

    ################
    # PEFT
    ################
    peft_config = get_peft_config(model_args)

    ################
    # Rendezvous
    ################
    rendezvous = Rendezvous(
        rendezvous_dir=script_args.rendezvous_dir,
        my_group_name=script_args.group,
    )

    ################
    # Training
    ################
    trainer = CoGRPOdp4RegimeTrainer(
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
        peft_config=peft_config,
        my_group_name=script_args.group,
        rendezvous=rendezvous,
        self_consistency_threshold=script_args.self_consistency_threshold,
        log_oracle_accuracy=script_args.log_oracle_accuracy,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
