set -xv
pip install wandb
pip install spacy
pip install nltk
wandb login 7e02057ca7216a04662368e6c2ee2eb4358b7762
accelerate launch --multi_gpu run_summarization_no_trainer.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir model/bart-base_cnn-dailymail_rouge1 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 20 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --weight_decay 0.01 \
    --seed 1111 \
    --checkpointing_steps epoch \
    --max_target_length 256 \
    --num_beams 4 \
    --preprocessing_num_workers 25 \
    --decoder_prompt svo \
    --with_tracking \
    --report_to wandb \
