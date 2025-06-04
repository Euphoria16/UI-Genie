#!/bin/bash
unset OMPI_COMM_WORLD_LOCAL_RANK
task_index=$OMPI_COMM_WORLD_RANK

MASTER=$(head -n 1 /job/hostfile | awk '{print $1}')
echo $MASTER
hostgpus=$(nvidia-smi -L|grep GPU|wc -l)

exp=agent_models/qwen2.5_vl_72b_rslora
model_path=/path/to/Qwen/Qwen2.5-VL-72B-Instruct
max_pixels=602112


exp_name="$exp"
mkdir -p output/"$exp_name"

MAX_PIXELS=$max_pixels \
NNODES=$(wc -l < /job/hostfile) \
NODE_RANK=$task_index \
MASTER_ADDR=$MASTER \
NPROC_PER_NODE=$hostgpus \
swift sft \
  --model "$model_path" \
  --dataset data/androidcontrol_train.jsonl data/androidlab_som_train.jsonl data/AMEX_Agent_34K.jsonl data/UI-Genie-Agent.jsonl \
  --train_type lora \
  --use_rslora True \
  --lora_rank 64 \
  --lora_alpha 256 \
  --target_modules all-linear \
  --init_weights pissa \
  --freeze_vit False \
  --freeze_aligner False \
  --freeze_llm False \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --max_pixels $max_pixels \
  --gradient_accumulation_steps 1 \
  --eval_steps 500 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --logging_steps 5 \
  --max_length 32768 \
  --output_dir output/"$exp_name" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --save_only_model True \
  --lr_scheduler_type cosine \
  --split_dataset_ratio 0 \
  --ddp_timeout 86400 \
  --deepspeed zero3_offload \
  --add_version False \
  2>&1 | tee -a output/"$exp_name"/output.log
