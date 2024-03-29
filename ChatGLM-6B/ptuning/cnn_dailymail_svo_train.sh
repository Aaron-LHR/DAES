CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --do_train \
    --do_eval \
    --train_file data/cnn_dailymail_svo_train.json \
    --validation_file data/cnn_dailymail_svo_test.json \
    --prompt_column article \
    --response_column highlights \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/chatglm-6b \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 256 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 3 \
    --predict_with_generate \
    --num_train_epochs 5 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --preprocessing_num_workers 10

