accelerate launch --multi_gpu --gpu_ids "1,2,3" main-accelerate_rouge_1.py \
--batch_size 36 \
--learning_rate 2e-3 \
--pretrained_model bert-base-uncased \
-a