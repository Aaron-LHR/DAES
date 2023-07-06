accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name cnn_dailymail \
    --use_cached_dataset '说' \
    --cached_dataset_path data/cnn-dailymail_cls \
    --filtered_dataset_path data/cnn-dailymail_filtered \
    --dataset_config "3.0.0" \
    --output_dir model/bart_base_cnn-dailymail_cls_step1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --weight_decay 0.01 \
    --seed 1111 \
    --checkpointing_steps epoch \
    --max_target_length 256 \
    --num_beams 4 \
    --preprocessing_num_workers 40 \
    --encoder_prompt multi_cls_step1 \
    --pad_to_max_length \
    --with_tracking \
    --report_to wandb