import os
import numpy as np
import torch
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification, BertModel
import math
import torch.nn as nn
import os
import numpy as np
import re
import matplotlib
import argparse
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
import copy
import math
from tqdm.auto import tqdm
from torch.nn import SyncBatchNorm
from torch.utils.data.dataloader import default_collate
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
import datasets
import evaluate
from datasets import load_dataset
work_dir = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("cnn_dailymail", '3.0.0')


from nltk.tokenize import sent_tokenize
sent_tokenize(dataset["train"][0]["article"])


def split_article(example):
    example["split_article"] = sent_tokenize(example["article"])
    example["split_highlights"] = sent_tokenize(example["highlights"])
    return example


split_dataset = dataset.map(split_article, num_proc=20, remove_columns=["article", "highlights"])

max_article_len = 0
for split_article in tqdm(split_dataset["train"]["split_article"]):
    max_article_len = max(max_article_len, len(split_article))
for split_article in tqdm(split_dataset["validation"]["split_article"]):
    max_article_len = max(max_article_len, len(split_article))
for split_article in tqdm(split_dataset["test"]["split_article"]):
    max_article_len = max(max_article_len, len(split_article))


# rouge = evaluate.load('rouge')
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
# results = rouge.compute(predictions=predictions, references=references)
# print(results)

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def get_extractive_summary(example):
    extractive_summary_label = [0 for _ in range(max_article_len)]
    extractive_summary_positions = set()
    for i, highlights_sent in enumerate(example["split_highlights"]):
        max_rouge_1 = 0
        max_rouge_1_position = None
        for j, article_sent in enumerate(example["split_article"]):
            if j not in extractive_summary_positions:
#                 rouge_scores = rouge.compute(predictions=[article_sent], references=[highlights_sent])
#                 rouge_1 = rouge_scores["rouge1"] 
                
                rouge_scores = scorer.score(article_sent, highlights_sent)
                rouge_1 = rouge_scores["rouge1"].fmeasure
                
                if rouge_1 > max_rouge_1:
                    max_rouge_1 = rouge_1
                    max_rouge_1_position = j
        if max_rouge_1_position is not None:
            extractive_summary_label[max_rouge_1_position] = 1
            extractive_summary_positions.add(max_rouge_1_position)
    example["extractive_summary_label"] = extractive_summary_label
    return example


rouge_1_dataset = split_dataset.map(get_extractive_summary, num_proc=100)

print("===================")
print("===================")
print("===================")
print("to save????")
rouge_1_dataset.save_to_disk("data/hugging_face_cnn_dailymail_rouge_1")
print("save done!!!!")
