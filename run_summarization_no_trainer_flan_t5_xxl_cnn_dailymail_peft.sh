accelerate launch --multi_gpu --gpu_ids "1,2" run_summarization_no_trainer.py \
    --model_name_or_path google/flan-t5-xxl \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir model/flan-t5-xxl_cnn-dailymail \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 14 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --weight_decay 0.01 \
    --seed 1111 \
    --checkpointing_steps epoch \
    --max_target_length 256 \
    --num_beams 4 \
    --preprocessing_num_workers 15 \
    --peft lora \
    --source_prefix summarize: \
    --with_tracking \
    --report_to wandb