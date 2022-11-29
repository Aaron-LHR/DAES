import os
import numpy as np
import torch
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification, BertModel
from accelerate import Accelerator, DistributedDataParallelKwargs
import math
from optparse import OptionParser
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
        

        
def string_collate(batch):
    text = {"src_txt": [], "tgt_txt": []}
    for d in batch:
        for k in text:
            text[k].append(d[k])
            del d[k]
        for k in d:
            d[k] = torch.tensor(d[k])
    return default_collate(batch), text
        
        
def test(model, options):
    rouge_metric = evaluate.load("rouge")
    model.eval()
    epoch_loss = 0
    for batch, text in test_data_loader:
        with torch.no_grad():
            outputs = model(batch)
        functioned_loss = loss_function(outputs, batch["labels"])
        masked_loss = functioned_loss * batch["labels_mask"]
        sumed_loss = masked_loss.sum()
        dived_loss = (sumed_loss / sumed_loss.numel())
        epoch_loss += dived_loss.item()

        for i in range(outputs.shape[0]):
            pred_label = outputs[i, :torch.nonzero(batch["labels_mask"][i]==0).squeeze()[0].item()] >= options.threshold
            pred_summarization_list = [text["src_txt"][i][j] for j, cls_pred in enumerate(pred_label) if cls_pred == True]
            pred_summarization = " ".join(pred_summarization_list)
            reference = text["tgt_txt"][i]
            accelerator.gather_for_metrics((pred_summarization, reference))
            rouge_metric.add_batch(predictions=[pred_summarization], references=[reference])
    
    
    accelerator.print("=============test=============")
    accelerator.print(f"rouge:")
    accelerator.print(rouge_metric.compute())
    accelerator.print(f'test loss: {epoch_loss}')        

        
def parse_param():
    usage = "training accerlate"
    parser = OptionParser(usage=usage, version="%prog 1.0")
    parser.add_option("-b", action="store", dest="batch_size", default=32, type=int, help='batch_size')
    parser.add_option("-l", action="store", dest="learning_rate", default=2e-3, type=float, help='learning_rate')
    parser.add_option("-p", action="store", dest="pretrained_model", default="bert-base-uncased", type=str, help='pretrained model')
    parser.add_option("-t", action="store", dest="threshold", default=0.5, type=float, help='threshold')
    
#     parser.add_option("-f", action="store_true", dest="reflection_service", help='whether to use reflection_service')
    options, args = parser.parse_args()
    return options, args, usage        
        
        
def main():
    distributedDataParallelKwargs = DistributedDataParallelKwargs()
    distributedDataParallelKwargs.find_unused_parameters = True
    accelerator = Accelerator(kwargs_handlers=[distributedDataParallelKwargs])
    device = accelerator.device
    accelerator.print("================")
    accelerator.print(f"device: {device}")
    
    # 解析参数
    options, args, usage = parse_param()
    
    work_dir = "./"

    model = BertSumExt(pretrained_model=options.pretrained_model)
    model.to(device)

    
    loss_function = torch.nn.BCELoss(reduction='none')
    learning_rate = options.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    

    final_dataset = load_from_disk("data/hugging_face_dataset_archived_bert_data_cnndm_final_sent")
    train_data_loader = torch.utils.data.DataLoader(final_dataset["train"], batch_size=options.batch_size, shuffle=False, collate_fn=string_collate)
    test_data_loader = torch.utils.data.DataLoader(final_dataset["test"], batch_size=options.batch_size, shuffle=False, collate_fn=string_collate)
    
    model, optimizer, train_data_loader, test_data_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_data_loader,
        test_data_loader,
        scheduler
    )
    
    accelerator.register_for_checkpointing(model, optimizer, scheduler)

    
    min_loss = None
    epochs = 1
    for i in tqdm(range(epochs), disable=not accelerator.is_local_main_process):
        epoch_loss = 0
        model.train()
        for batch, text in tqdm(train_data_loader, disable=not accelerator.is_local_main_process):
            optimizer.zero_grad()
            
            outputs = model(batch)


            functioned_loss = loss_function(outputs, batch["labels"])
            masked_loss = functioned_loss * batch["labels_mask"]
            sumed_loss = masked_loss.sum()
            dived_loss = (sumed_loss / sumed_loss.numel())
            iter_loss = dived_loss
            epoch_loss += dived_loss.item()

            accelerator.backward(iter_loss)
            
#             for name, param in model.named_parameters():
#                 if param.grad is None:
#                     accelerator.print(name + "\t======")
#                 else:
#                     accelerator.print(name)
            
            optimizer.step()

        if i % 1 == 0:
            accelerator.print(f'epoch: {i} loss: {epoch_loss: .4f}\tlr: {optimizer.param_groups[0]["lr"]: .10f}')
            
        if i % 1 == 0:
            scheduler.step(epoch_loss)
            
        if min_loss is None or epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model_path = f'{work_dir}model/BertSumExt-learn_rate_{learning_rate}-bce_loss-adam-epoch_{i}-e_loss_{epoch_loss}-last_lr_{optimizer.param_groups[0]["lr"]}'
            accelerator.save_state(model, best_model_path)
            
        test(model, options)
        
if __name__ == "__main__":
    main()