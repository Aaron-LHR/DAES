from datasets import load_dataset, load_from_disk


rouge_1_dataset = load_from_disk("./data/hugging_face_cnn_dailymail")
print(rouge_1_dataset)

for key in rouge_1_dataset.keys():
    rouge_1_dataset[key].push_to_hub("Aaron-LHR/cnn_dailymail_extractive_summary_rouge_1", split=key)