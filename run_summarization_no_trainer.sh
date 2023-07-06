accelerate launch  run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-base \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir model/mbart-large-50-summarization \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --lr_scheduler_type 'cosine' \
    --num_warmup_steps 10000 \
    --seed 1111 \
    --checkpointing_steps epoch \
    --max_length 1024 \
    --max_target_length 128 \
    --num_beams 1 \
#     --with_tracking