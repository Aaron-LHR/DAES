# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 run_summarization.py \
python run_summarization.py \
    --model_name_or_path google/mt5-base \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir model/mbart-large-50-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
