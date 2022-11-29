import os
import numpy as np
import torch
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification, BertModel
import math
import torch.nn as nn
import os
import numpy as np
import matplotlib
import argparse
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
import copy
import math
from tqdm.auto import tqdm
from torch.nn import SyncBatchNorm
from torch.utils.data.dataloader import default_collate
from torch.nn.parallel import DistributedDataParallel
from datasets import load_from_disk
import torch.distributed as dist
import evaluate
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
work_dir = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("================")
print(f"device: {device}")

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--local_rank', type=int, help='local rank for dist')
parser.add_argument('--batch_size', type=int, help='local rank for dist', default=32)

args = parser.parse_args()

print(os.environ['MASTER_ADDR'])
print(os.environ['MASTER_PORT'])
world_size = torch.cuda.device_count()
local_rank = args.local_rank

dist.init_process_group(backend='nccl')

torch.cuda.set_device(local_rank)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]
    

class BertSumExt(torch.nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", nhead=8, transformer_layers=2, dropout=0.1):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model)
        d_model = self.bert.config.hidden_size
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_encode_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encode_layer, num_layers=transformer_layers)
        self.liner = nn.Linear(d_model, 1, bias=True)
        
    
    def forward(self, inputs):
#         last_hidden_state = self.bert(**inputs).last_hidden_state
        last_hidden_state = self.bert(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
#         last_hidden_state = self.bert(**{k: inputs[k].to(device) for k in ['input_ids', 'token_type_ids', 'attention_mask']}).last_hidden_state
#         print(last_hidden_state.shape)
#         print(inputs["clss"].shape)
        cls_embeddings = last_hidden_state[torch.arange(last_hidden_state.size(0)).unsqueeze(1), inputs["clss"]]
        
        input_transformer = cls_embeddings + self.pos_emb.pe[:, cls_embeddings.shape[1]]
        
#         print(cls_embeddings.shape)
#         print(input_transformer.shape)
#         print(inputs["clss_mask"].shape)
        transformer_embeddings = self.transformer_encoder(cls_embeddings, src_key_padding_mask=inputs["labels_mask"]==0)
        liner_embeddings = self.liner(transformer_embeddings)
        sigmoid_embeddings = torch.sigmoid(liner_embeddings)
        final_embeddings = sigmoid_embeddings.squeeze(-1)
#         print(transformer_embeddings.shape)
#         print(liner_embeddings.shape)
#         print(sigmoid_embeddings.shape)
#         print(final_embeddings.shape)
        
        return final_embeddings
        
        
        
pretrained_model="bert-base-uncased"

model = BertSumExt(pretrained_model="bert-base-uncased")


model = model.cuda()
model = SyncBatchNorm.convert_sync_batchnorm(model)
model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

loss_function = torch.nn.BCELoss(reduction='none')
learning_rate = 2e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)


def string_collate(batch):
    text = {"src_txt": [], "tgt_txt": []}
    for d in batch:
        for k in text:
            text[k].append(d[k])
            del d[k]
        for k in d:
            d[k] = torch.tensor(d[k])
    return default_collate(batch), text


final_dataset = load_from_disk("data/hugging_face_dataset_archived_bert_data_cnndm_final")
final_dataset.set_format(type="torch")
train_sampler = DistributedSampler(final_dataset["train"], shuffle=True)
train_data_loader = torch.utils.data.DataLoader(final_dataset["train"], sampler=train_sampler, batch_size=args.batch_size, shuffle=False)
# test_sampler = DistributedSampler(final_dataset["test"], shuffle=True)
# test_data_loader = torch.utils.data.DataLoader(final_dataset["test"], sampler=test_sampler, batch_size=args.batch_size, shuffle=False, collate_fn=string_collate)

    
rouge_metric = evaluate.load("rouge")
def test(model):
    model.eval()
    test_loss = 0
    if local_rank == 0:
        bar = tqdm(test_data_loader)
    for batch, text in test_data_loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(batch)
        functioned_loss = loss_function(outputs, batch["labels"])
        masked_loss = functioned_loss * batch["labels_mask"]
        sumed_loss = masked_loss.sum()
        dived_loss = (sumed_loss / sumed_loss.numel())
        test_loss += dived_loss.item()

        for i in range(outputs.shape[0]):
            pred_label = outputs[i, :torch.nonzero(batch["labels_mask"][i]==0).squeeze()[0].item()] >= 0.5
            pred_summarization_list = [text["src_txt"][i][j] for j, cls_pred in enumerate(pred_label) if cls_pred == True]
            pred_summarization = " ".join(pred_summarization_list)
            reference = text["tgt_txt"][i]
            rouge_metric.add_batch(predictions=[pred_summarization], references=[reference])
            
        if local_rank == 0:
            bar.update(1)
            
    if local_rank == 0:
        print("=============test=============")
        print(f"rouge:")
        print(rouge_metric.compute())
        print(f'test loss: {test_loss: .4f}')

    
min_loss = 10
epochs = 50
if local_rank == 0:
    bar = tqdm(range(epochs))
    
for i in tqdm(range(epochs), disable=local_rank!=0):
    epoch_loss = 0
    model.train()
#     train_data_loader.sampler.set_epoch(i)
#     test_data_loader.sampler.set_epoch(i)
    for batch in tqdm(train_data_loader, disable=local_rank!=0):
        optimizer.zero_grad()
#         for k in ['input_ids', 'token_type_ids', 'attention_mask', 'clss', 'clss_mask', "labels"]:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(batch)


        single_loss = loss_function(outputs, batch["labels"])
#         print(single_loss.shape)
#         print(single_loss)
        
        single_loss = single_loss * batch["labels_mask"]
#         print(single_loss.shape)
#         print(single_loss)
        single_loss = single_loss.sum()
#         print(single_loss)
        single_loss = (single_loss / single_loss.numel())
#         print(single_loss)
        epoch_loss += single_loss.item()
        single_loss.backward()
        optimizer.step()
#         break
    if i % 1 == 0 and local_rank == 0:
        print(f'epoch: {i} loss: {epoch_loss: .4f}\tlr: {optimizer.param_groups[0]["lr"]: .10f}')
        bar.update(1)
    if i % 1 == 0:
        scheduler.step(epoch_loss)
    if epoch_loss < min_loss and local_rank == 0:
        min_loss = epoch_loss
        best_model_path = f'{work_dir}model/BertSumExt-learn_rate_{learning_rate}-bce_loss-adam-epoch_{i}-e_loss_{epoch_loss}-last_lr_{optimizer.param_groups[0]["lr"]: .10f}.chpk'
        torch.save(model, best_model_path)
#     if i % 1 == 0:
#         del batch
#         torch.cuda.empty_cache() # 释放显存
#         test(model)

        