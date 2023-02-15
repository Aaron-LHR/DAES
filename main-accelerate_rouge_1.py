import os
import numpy as np
import torch
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification, BertModel, BertConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
import math
from optparse import OptionParser
import torch.nn as nn
import os
import numpy as np
# import matplotlib
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
from datasets import load_from_disk
import datasets
import torch.distributed as dist
import evaluate
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
from config_map import config_map
from rouge_score import rouge_scorer
import random


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
        config = BertConfig.from_json_file("bert_position_embedding_config.json")
        self.bert = BertModel.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
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
    text = {"id": [], "split_article": [], "split_highlights": []}
    for d in batch:
        for k in text:
            text[k].append(d[k])
            del d[k]
        for k in d:
            d[k] = torch.tensor(d[k])
    return default_collate(batch), text



def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False
 
 
        
def test(model, options, test_data_loader, loss_function, accelerator, tokenizer):
    accelerator.print("-" * 10 + "test" + "-" * 10)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    model.eval()
    epoch_loss = 0
    rouge_score_list = []
    with torch.no_grad():
        for batch, text in tqdm(test_data_loader, disable=not accelerator.is_local_main_process):
            for k, v in batch.items():
                batch[k] = v.to(accelerator.device)
            outputs = model(batch)
            functioned_loss = loss_function(outputs, batch["extractive_summary_label"] * 1.0)
            masked_loss = functioned_loss * batch["labels_mask"]
            sumed_loss = masked_loss.sum()
            dived_loss = (sumed_loss / sumed_loss.numel())
            epoch_loss += dived_loss.item()


            # todo:
            # 1.tokenize之后gather
            # 2.单进程
            for i in range(outputs.shape[0]):
                pred_label = outputs[i, :torch.nonzero(batch["labels_mask"][i]==0).squeeze()[0].item()] >= options.threshold
                pred_summarization_list = [text["split_article"][i][j] for j, cls_pred in enumerate(pred_label) if cls_pred == True]
                pred_summarization = " ".join(pred_summarization_list)
                reference = " ".join(text["split_highlights"][i])
                rouge_score_list.append(scorer.score(pred_summarization, reference)["rouge1"].fmeasure)
    #             accelerator.gather_for_metrics((pred_summarization, reference))
    #             accelerator.print("=============predictions=============")
    #             accelerator.print(type(predictions))
    #             accelerator.print(predictions)
    #             accelerator.print("=============references=============")
    #             accelerator.print(type(references))
    #             accelerator.print(references)

    #             print("=============pred_summarization=============")
    #             print(type(pred_summarization))
    #             print(pred_summarization)
    #             print("=============reference=============")
    #             print(type(reference))
    #             print(reference)
    #             rouge_metric.add({'predictions': pred_summarization, 'references': reference})
    #             rouge_metric.add(predictions=pred_summarization, references=reference)
    #             if accelerator.is_local_main_process:
    #                 predictions.append("a")
    #                 references.append("a")
    #             else:
    #             predictions.append(pred_summarization)
    #             references.append(reference)

    #         accelerator.gather_for_metrics((predictions, references))
    #             accelerator.print(predictions)
    #         rouge_metric.add_batch(predictions=predictions, references=references)
#             break
    
    
#     accelerator.wait_for_everyone()
#     rouge_score = rouge_metric.compute()
    rouge_score_list = torch.Tensor(rouge_score_list).to(accelerator.device)
    rouge_score_list = accelerator.gather_for_metrics(rouge_score_list)
    
    mean_rouge_score = torch.mean(rouge_score_list)
    accelerator.print(f"rouge:{mean_rouge_score}")
#     accelerator.print(rouge_metric.compute(predictions=predictions, references=references))
    accelerator.print(f'test loss: {epoch_loss}')
    return epoch_loss, mean_rouge_score

        
def parse_param():
    usage = "training accerlate"
    parser = OptionParser(usage=usage, version="%prog 1.0")
    parser.add_option("-b", "--batch_size", action="store", dest="batch_size", default=36, type=int, help='batch_size')
    parser.add_option("-l", "--learning_rate", action="store", dest="learning_rate", default=2e-3, type=float, help='learning_rate')
    parser.add_option("-p", "--pretrained_model", action="store", dest="pretrained_model", default="bert-base-uncased", type=str, help='pretrained model')
    parser.add_option("-t", action="store", dest="threshold", default=0.5, type=float, help='threshold')
    parser.add_option("-s", action="store", dest="random_seed", default=1111, type=int, help='random_seed')
    parser.add_option("-o", action="store", dest="optimizer", default="Adam", type=str, help='optimizer')
    parser.add_option("-r", action="store", dest="scheduler", default="ReduceLROnPlateau", type=str, help='scheduler')
    parser.add_option("-a", action="store_true", dest="save_per_epoch", help='whether to save_per_epoch')
    options, args = parser.parse_args()
    return options, args, usage        
        
        
def main():
    distributedDataParallelKwargs = DistributedDataParallelKwargs()
    distributedDataParallelKwargs.find_unused_parameters = True
    accelerator = Accelerator(kwargs_handlers=[distributedDataParallelKwargs])
    device = accelerator.device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator.print("================")
    accelerator.print(f"device: {device}")
    
    # 解析参数
    options, args, usage = parse_param()
    
    work_dir = "./"
    random.seed(options.random_seed)
    torch.manual_seed(options.random_seed)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertSumExt(pretrained_model=options.pretrained_model)
    model.to(device)

    loss_function = torch.nn.BCELoss(reduction='none')
    optimizer, scheduler = config_map(options, model)
    

    final_dataset = load_from_disk("data/hugging_face_cnn_dailymail_rouge_1_for_train")
    train_data_loader = torch.utils.data.DataLoader(final_dataset["train"], batch_size=options.batch_size, shuffle=False, collate_fn=string_collate)
    test_data_loader = torch.utils.data.DataLoader(final_dataset["test"], batch_size=options.batch_size, shuffle=False, collate_fn=string_collate)
    
    model, optimizer, train_data_loader, test_data_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_data_loader,
        test_data_loader,
        scheduler
    )
    
#     model, optimizer, train_data_loader, scheduler = accelerator.prepare(
#         model,
#         optimizer,
#         train_data_loader,
#         scheduler
#     )
    
    accelerator.register_for_checkpointing(model, optimizer, scheduler)
    model_path = f'{work_dir}model/BertSumExt-learningrate_{options.learning_rate}-loss_bce-optimizer_{options.optimizer}-bcs_{options.batch_size}-scheduler_{options.scheduler}-randomseed_{options.random_seed}'
    if accelerator.is_local_main_process:
        mkdir(model_path)
    
    min_eloss = None
    min_tloss = None
    max_rouge = None
    best_model_path = None
    epochs = 1
    for i in tqdm(range(epochs), disable=not accelerator.is_local_main_process):
        accelerator.print(f"===========epoch:{i}===========")
        accelerator.print("-" * 10 + "train" + "-" * 10)
        epoch_loss = 0
        save_flag = False
        model.train()
        for batch, text in tqdm(train_data_loader, disable=not accelerator.is_local_main_process):
            optimizer.zero_grad()
            
            outputs = model(batch)


            functioned_loss = loss_function(outputs, batch["extractive_summary_label"] * 1.0)
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
#             break

        if i % 1 == 0:
            accelerator.print(f'epoch: {i} loss: {epoch_loss: .4f}\tlr: {optimizer.param_groups[0]["lr"]: .10f}')
            
        if i % 1 == 0:
            scheduler.step(epoch_loss)
            
#         accelerator.wait_for_everyone()
#         if accelerator.is_local_main_process:
        test_loss, rouge_score = test(model, options, test_data_loader, loss_function, accelerator, tokenizer)
        
        if min_eloss is None or epoch_loss < min_eloss:
            min_eloss = epoch_loss
            save_flag = True

        if min_tloss is None or test_loss < min_tloss:
            min_tloss = test_loss
            save_flag = True
            best_save_path = os.path.join(model_path, f'best_epoch_{i}-eloss_{epoch_loss}-tloss_{test_loss}-rouge_{str(rouge_score)}-lr_{optimizer.param_groups[0]["lr"]}')


        if max_rouge is None or rouge_score >= max_rouge:
            max_rouge = rouge_score
            save_flag = True
            best_save_path = os.path.join(model_path, f'best_epoch_{i}-eloss_{epoch_loss}-tloss_{test_loss}-rouge_{str(rouge_score)}-lr_{optimizer.param_groups[0]["lr"]}')

        if options.save_per_epoch:
            save_flag = True

        if save_flag and accelerator.is_local_main_process:
            save_path = f'epoch_{i}-eloss_{epoch_loss}-tloss_{test_loss}-rouge_{str(rouge_score)}-lr_{optimizer.param_groups[0]["lr"]}'
            accelerator.save_state(os.path.join(model_path, save_path))
        accelerator.wait_for_everyone()
        
    if accelerator.is_local_main_process:
        accelerator.save_state(best_save_path)
        
        
if __name__ == "__main__":
    main()