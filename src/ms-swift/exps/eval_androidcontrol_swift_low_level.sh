#!/bin/bash



DATA_PATH=data/androidcontrol_test_low_level.jsonl

MODEL_PATH=/path/to/UI-Genie-7B-Agent

expname=UI-Genie-7B_androidcontrol
SAVE_NAME=androidcontrol_infer_json/low_level_$expname.jsonl
echo "$SAVE_NAME"
max_pixels=602112

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_PIXELS=$max_pixels \
swift infer \
    --model $MODEL_PATH \
    --model_type qwen2_5_vl \
    --stream false \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 256\
    --val_dataset $DATA_PATH \
    --result_path $SAVE_NAME \
    --infer_backend "pt" \
    --max_pixels $max_pixels    \

python eval_results_androidcontrol.py --pred_name $SAVE_NAME --input_name $DATA_PATH --max_pixels $max_pixels --model_path $MODEL_PATH
