wandb offline
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/grpo/accelerate_lora.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 4 \
    --main_process_port 19346 \
    projects/grpo/train_grpo.py \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --model_name_or_path Qwen/Qwen3-4B \
    --output_dir projects/grpo/work_dirs/qwen4b-1epoch-lora-r16-alpha32-opsd30k_bs1_acc4_lr2e-5_gen8_temp1.2/ \
    --train_dataset siyanzhao/Openthoughts_math_30k_opsd \
    --run_config qwen4b-1epoch-lora-r16-alpha32-opsd30k \
    --num_train_epochs 1 \
    --report_to wandb \
    --gradient_checkpointing \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_completion_length 16000 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --use_peft \
    --vllm_mode colocate \
    --vllm_max_model_length 4096 \
    --logging_steps 10 \
    --save_steps 200 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project GRPO
