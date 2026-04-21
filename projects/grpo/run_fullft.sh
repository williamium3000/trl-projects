wandb offline
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MATH500_EVAL_PATH=data/math500/test.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/grpo/accelerate.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 4 \
    --main_process_port 19346 \
    projects/grpo/train_grpo.py \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --model_name_or_path Qwen/Qwen3-4B \
    --output_dir projects/work_dirs/grpo/qwen4b-1epoch-fullft-opsd30k_bs1_acc4_lr2e-5_gen8_temp1.2/ \
    --train_dataset siyanzhao/Openthoughts_math_30k_opsd \
    --run_config qwen4b-1epoch-fullft-opsd30k \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --max_completion_length 16000 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 4096 \
    --logging_steps 10 \
    --save_steps 200 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project GRPO
