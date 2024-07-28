#!/usr/bin/env bash
TASK_NAME=laptop14
python main_t5encoder.py --model_type t5 \
            --task_name ${TASK_NAME} \
            --data_dir ./data/uabsa/${TASK_NAME} \
            --model_name_or_path ./pretrained-models/t5-base \
            --checkpoint checkpoint-g \
            --do_train \
            --do_eval \
            --per_gpu_train_batch_size 32 \
            --gradient_accumulation_steps 2 \
            --per_gpu_eval_batch_size 32 \
            --overwrite_output_dir \
            --tagging_schema OT \
            --learning_rate 3e-5 \
            --save_steps 100 \
            --max_steps 4000 \
            --device 1
