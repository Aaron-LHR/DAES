#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
from collections import defaultdict, Counter
import platform
# if "Windows" in platform.system() or "windows" in platform.system():
#     os.environ["http_proxy"] = "http://127.0.0.1:7890"
#     os.environ["https_proxy"] = "http://127.0.0.1:7890"
import random
import unicodedata
from pathlib import Path
from nltk.tokenize import sent_tokenize
import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import traceback
import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_TYPES = tuple(list(MODEL_TYPES) + ["BartForConditionalGenerationWithRouge1", "BartForConditionalGenerationWithRouge1Rouge2"])

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--target_domain_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--target_domain_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--target_domain_samples_num",
        type=int,
        default=None,
        help="The number of samples of the target domain dataset to use.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", default=False, action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
        help="peft options.",
    )
    parser.add_argument(
        "--decoder_prompt",
        type=str,
        default=None,
        help="decoder prompt options.",
    )
    parser.add_argument(
        "--encoder_prompt",
        type=str,
        default=None,
        help="encoder prompt options.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only do test",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=0,
        help="resume_step",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    from bart_with_ext import BartForConditionalGenerationWithRougeClass, BartForConditionalGenerationWithRouge1, BartForConditionalGenerationWithRouge1Rouge2
    from DataCollatorForBartForConditionalGenerationWithRouge import DataCollatorForBartForConditionalGenerationWithRouge

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if args.target_domain_dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        target_domain_raw_datasets = load_dataset(args.target_domain_dataset_name, args.target_domain_dataset_config_name)
        if args.target_domain_samples_num is not None and args.target_domain_samples_num != 0:
            target_domain_raw_datasets["train"] = target_domain_raw_datasets["train"].train_test_split(train_size=args.target_domain_samples_num)["train"]

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.model_type == "BartForConditionalGenerationWithRouge1":
            model = BartForConditionalGenerationWithRouge1.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
        elif args.model_type == "BartForConditionalGenerationWithRouge1Rouge2":
            model = BartForConditionalGenerationWithRouge1Rouge2.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
        elif args.peft == "lora":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

            # Define LoRA Config
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            # prepare int-8 model for training
            model = prepare_model_for_int8_training(model)

            # add LoRA adaptor
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    def clean(example):
        example[text_column] = unicodedata.normalize('NFKC', example[text_column])
        example[text_column] = ' '.join([x for x in example[text_column].split(' ') if x != ''])

        example[summary_column] = unicodedata.normalize('NFKC', example[summary_column])
        example[summary_column] = ' '.join([x for x in example[summary_column].split(' ') if x != ''])
        return example

    accelerator.print(raw_datasets)
    raw_datasets = raw_datasets.map(clean)
    raw_datasets = raw_datasets.filter(lambda example: len(sent_tokenize(example[text_column])) > 3)
    accelerator.print(raw_datasets)

    if args.target_domain_dataset_name is not None:
        target_domain_dataset_columns = summarization_name_mapping.get(args.target_domain_dataset_name, None)
        target_domain_raw_datasets = target_domain_raw_datasets.rename_column(target_domain_dataset_columns[0], text_column)
        target_domain_raw_datasets = target_domain_raw_datasets.rename_column(target_domain_dataset_columns[1], summary_column)

        # for split in target_domain_raw_datasets.keys():
        #     target_domain_raw_datasets[split][text_column] = target_domain_raw_datasets[split][target_domain_dataset_columns[0]]
        #     target_domain_raw_datasets[split][summary_column] = target_domain_raw_datasets[split][target_domain_dataset_columns[1]]
        #     target_domain_raw_datasets[split] = target_domain_raw_datasets[split].remove_columns(target_domain_dataset_columns)

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def get_salient_score_rouge1(article, summary):
        scores = {}
        AS = len([token for token in article if token != tokenizer.pad_token_id])
        SS = len([token for token in summary if token != tokenizer.pad_token_id])
        a_counter = Counter(article)
        s_counter = defaultdict(int)
        for k, v in Counter(summary).items():
            s_counter[k] = v
        res = []
        for token in article:
            if token not in scores:
                if token in summary:
                    scores[token] = 1 - np.exp(-(np.log(AS / a_counter[token]) / np.log(SS / s_counter[token])))
                else:
                    scores[token] = 0
            res.append(scores[token])
        return res, scores

    def get_input_sailent_scores(article, summary):
        scores = get_salient_score_rouge1(article, summary)
        # return torch.Tensor([scores[token] if token not in tokenizer.all_special_ids else 0 for token in article])
        return torch.Tensor([scores[token] for token in article])

    def get_batch_sailent_scores_rouge1(articles, summaries):
        assert len(articles) == len(summaries)
        batch_scores = []
        for i in range(len(articles)):
            article = articles[i]
            summary = summaries[i]
            article_scores, scores = get_salient_score_rouge1(article, summary)
            # batch_scores.append([scores[token] if token not in tokenizer.all_special_ids else -100 for token in article])
            batch_scores.append(article_scores)
        return batch_scores

    def get_salient_score_rouge2(article, summary):
        scores = {}
        AS = len([token for token in article if token != tokenizer.pad_token_id]) - 1
        SS = len([token for token in summary if token != tokenizer.pad_token_id]) - 1
        article_2_grams = [f"{article[i]} {article[i + 1]}" for i in range(AS)]
        summary_2_grams = [f"{summary[i]} {summary[i + 1]}" for i in range(SS)]
        a_counter = Counter(article_2_grams)
        s_counter = defaultdict(int)
        for k, v in Counter(summary_2_grams).items():
            s_counter[k] = v
        res = []
        for gram_2 in article_2_grams:
            if gram_2 not in scores:
                if gram_2 in summary_2_grams:
                    scores[gram_2] = 1 - np.exp(-(np.log(AS / a_counter[gram_2]) / np.log(SS / s_counter[gram_2])))
                else:
                    scores[gram_2] = 0
            res.append(scores[gram_2])
        return res, scores

    def get_batch_sailent_scores_rouge2(articles, summaries):
        assert len(articles) == len(summaries)
        batch_scores = []
        for i in range(len(articles)):
            article = articles[i]
            summary = summaries[i]
            article_scores, scores = get_salient_score_rouge2(article, summary)
            batch_scores.append(article_scores)
        return batch_scores

    def get_rouge_prompt_article(example):
        from nltk.tokenize import sent_tokenize
        article_sents = sent_tokenize(example['article'])
        highlights_sents = sent_tokenize(example['highlights'])
        if len(highlights_sents) == 1:
            highlights_sents = sent_tokenize(example['highlights'].replace("\n", ". "))
        if len(highlights_sents) == 1:
            highlights_sents = sent_tokenize(example['highlights'].replace("  ", ". "))

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        for i, article_sent in enumerate(article_sents):
            rouges = []
            for j, highlights_sent in enumerate(highlights_sents):
                score = scorer.score(highlights_sent, article_sent)["rouge2"].fmeasure
                rouges.append((j + 1, score))
            rouges = sorted(rouges, key=lambda x: x[1], reverse=True)
            if rouges[0][1] > 0:
                article_sents[i] = f'[{rouges[0][0]}] {article_sents[i]} [/{rouges[0][0]}]'
        example['article'] = ' '.join(article_sents)
        return example

    punctuations = [',', '.', ';', '"', "'", '?', '!', ':']
    def preprocess_function(examples):
        inputs = examples[text_column]
        for punctuation in punctuations:
            inputs = [inp.replace(f'{punctuation} ', f'{punctuation}').replace(f'{punctuation}', f'{punctuation} ') for inp in inputs]
        inputs = [' ' + inp.lstrip() for inp in inputs]
        targets = examples[summary_column]
        targets = [x.replace("\n", " ") for x in targets]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        if args.model_type == "BartForConditionalGenerationWithRouge1":
            model_inputs["ext_rouge1_labels"] = get_batch_sailent_scores_rouge1(model_inputs["input_ids"], model_inputs["labels"])
        elif args.model_type == "BartForConditionalGenerationWithRouge1Rouge2":
            model_inputs["ext_rouge1_labels"] = get_batch_sailent_scores_rouge1(model_inputs["input_ids"], model_inputs["labels"])
            model_inputs["ext_rouge2_labels"] = get_batch_sailent_scores_rouge2(model_inputs["input_ids"], model_inputs["labels"])
        return model_inputs

    def cluster_prompt(batch, outputs):
        from sklearn.cluster import KMeans
        from nltk.tokenize import sent_tokenize
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        articles = tokenizer.batch_decode(batch.input_ids, skip_special_tokens=True)
        for punctuation in punctuations:
            articles = [inp.replace(f'{punctuation} ', f'{punctuation}').replace(f'{punctuation}', f'{punctuation} ') for inp in articles]
        for article_index, article in enumerate(articles):
            sents = sent_tokenize(article)
            kmeans = KMeans(n_clusters=3)
            embeddings = []
            start_position = 1
            for sent_id, sent in enumerate(sents):
                if sent_id != 0:
                    sent = ' ' + sent
                sent_len = len(tokenizer(sent).input_ids) - 2
                sentence_embedding = encoder_last_hidden_state[article_index, start_position: start_position + sent_len]
                embeddings.append(torch.mean(sentence_embedding, dim=0))
                start_position += sent_len
            embeddings = torch.stack(embeddings)
            # print(embeddings.shape)
            np_emb = embeddings.cpu().detach().numpy()
            try:
                kmeans = kmeans.fit(np_emb)
            except:
                traceback.print_exc()
                start_position = 1
                for sent_id, sent in enumerate(sents):
                    if sent_id != 0:
                        sent = ' ' + sent
                    sent_len = len(tokenizer(sent).input_ids) - 2
                    sentence_embedding = encoder_last_hidden_state[article_index, start_position: start_position + sent_len]
                    print('sentence_embedding')
                    print(sentence_embedding)
                    print(sentence_embedding.size())
                    print(sent)
                    print(tokenizer(sent).input_ids)
                    print(sent_len)
                    print(batch.input_ids[article_index, start_position: start_position + sent_len])
                    print(tokenizer.decode(batch.input_ids[article_index, start_position: start_position + sent_len], skip_special_tokens=False))
                    start_position += sent_len
                print(batch.input_ids[article_index])
                print(np_emb)
                print(embeddings)
                print(encoder_last_hidden_state[article_index, : start_position])

            # km_labels = km.labels_
            # print(km_labels)

            for n_cluster_index in range(kmeans.n_clusters):
                distances = kmeans.transform(np_emb)[:, n_cluster_index]
                # print(len([closest_i for closest_i in np.argsort(distances) if kmeans.labels_[closest_i] == n_cluster_index]))
                # closest = [closest_i for closest_i in np.argsort(distances) if kmeans.labels_[closest_i] == n_cluster_index][:3]
                closest = np.argsort(distances)[:3]
                for center_sent_index in closest:
                    sents[
                        center_sent_index] = f'[{n_cluster_index + 1}] {sents[center_sent_index]} [/{n_cluster_index + 1}]'
            article = ' '.join(sents)
            articles[article_index] = article
        return articles

    def batch_map_all_subject_verb_obj(examples):
        import spacy
        from nltk.tokenize import sent_tokenize
        # Load the parser
        nlp = spacy.load('en_core_web_sm')

        def get_all_subject_verb_object(sentence):
            """
            Extracts subject, verb and object from sentence using spaCy dependency parser.
            """
            # Parse the sentence
            doc = nlp(sentence)

            # Extract subject, verb and object
            subject = []
            verb = []
            obj = []

            for token in doc:
                if 'subj' in token.dep_:
                    subject.append(token.text)
                elif 'obj' in token.dep_:
                    obj.append(token.text)
                elif 'ROOT' in token.dep_:
                    verb.append(token.text)

            return subject, verb, obj

        def get_subject_verb_obj_new_label(sents):
            subjects, verbs, objs = [], [], []
            for sent in sent_tokenize(sents):
                subject, verb, obj = get_all_subject_verb_object(sent)
                subjects.extend(subject[:2])
                verbs.extend(verb[:2])
                objs.extend(obj[:2])
            # subjects = random.sample(subjects, min(len(subjects), 6))
            # verbs = random.sample(verbs, min(len(verbs), 6))
            # objs = random.sample(objs, min(len(objs), 6))
            res = f"Subjects: {', '.join(subjects)}. Predicates: {', '.join(verbs)}. Objects: {', '.join(objs)}. {prompt_sep_token} {sents}"
            return res

        summarys = []
        for summary in examples[summary_column]:
            summarys.append(get_subject_verb_obj_new_label(summary))
        return {summary_column: summarys}

    with accelerator.main_process_first():
        if args.decoder_prompt == "svo":
            prompt_sep_token = "Summary:"
            # tokenizer.add_special_tokens({"additional_special_tokens": [prompt_sep_token]})
            # prompt_sep_token_id = tokenizer.convert_tokens_to_ids(prompt_sep_token)
            raw_datasets = raw_datasets.map(
                batch_map_all_subject_verb_obj,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=True,
                desc="Running svo on dataset",
            )

        if args.encoder_prompt == "rouge":
            raw_datasets = raw_datasets.map(
                get_rouge_prompt_article,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=True,
                desc="Running knn prompt on dataset",
            )

        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        if args.target_domain_dataset_name is not None:
            processed_target_domain_datasets = target_domain_raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on target domain dataset",
            )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]
    if args.target_domain_dataset_name is not None:
        if args.target_domain_samples_num is None or args.target_domain_samples_num != 0:
            train_dataset = concatenate_datasets([train_dataset, processed_target_domain_datasets["train"]])

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.model_type in BartForConditionalGenerationWithRougeClass:
        data_collator = DataCollatorForBartForConditionalGenerationWithRouge(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    if args.target_domain_dataset_name is not None:
        target_domain_eval_dataloader = DataLoader(processed_target_domain_datasets["test"], collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.peft == "lora":
        from peft.utils.other import fsdp_auto_wrap_policy
        if getattr(accelerator.state, "fsdp_plugin", None) is not None:
            accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    if args.target_domain_dataset_name is not None:
        model, optimizer, train_dataloader, eval_dataloader, target_domain_eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, target_domain_eval_dataloader, lr_scheduler
        )
    else:
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if args.test:
        args.num_train_epochs = 1

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    # Metric
    metric = evaluate.load("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    debug_file = open("log", "w", encoding="utf-8")
    for epoch in range(starting_epoch, args.num_train_epochs):
        if not args.test:
            model.train()
            if args.with_tracking:
                total_loss = 0
                total_ext_rouge1_loss = 0
                total_ext_rouge2_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                if completed_steps < args.resume_step:
                    completed_steps += 1
                    continue

                with accelerator.accumulate(model):
                    outputs = model(**batch)

                    if args.encoder_prompt == "kmeans":
                        batch = batch.to('cpu')
                        inputs = cluster_prompt(batch, outputs)
                        del outputs
                        inputs = [prefix + inp for inp in inputs]
                        model_inputs = [tokenizer(input_, max_length=args.max_source_length, padding=padding, truncation=True) for input_ in inputs]
                        features = data_collator(model_inputs)
                        for key in ['input_ids', 'attention_mask']:
                            batch[key] = features[key]
                        batch = batch.to(accelerator.device)
                        outputs = model(**batch)
                        batch = batch.to('cpu')
                        del batch

                    if args.model_type == "BartForConditionalGenerationWithRouge1":
                        abs_loss = outputs.loss
                        masked_ext_rouge1_loss = outputs.masked_ext_rouge1_loss
                        # We keep track of the loss at each epoch
                        if args.with_tracking:
                            total_loss += abs_loss.detach().float()
                            total_ext_rouge1_loss += masked_ext_rouge1_loss.detach().float()
                        loss = abs_loss + masked_ext_rouge1_loss
                    elif args.model_type == "BartForConditionalGenerationWithRouge1Rouge2":
                        abs_loss = outputs.loss
                        masked_ext_rouge1_loss = outputs.masked_ext_rouge1_loss
                        masked_ext_rouge2_loss = outputs.masked_ext_rouge2_loss
                        # We keep track of the loss at each epoch
                        if args.with_tracking:
                            total_loss += abs_loss.detach().float()
                            total_ext_rouge1_loss += masked_ext_rouge1_loss.detach().float()
                            total_ext_rouge2_loss += masked_ext_rouge2_loss.detach().float()
                        loss = abs_loss + masked_ext_rouge1_loss + masked_ext_rouge2_loss
                    else:
                        loss = outputs.loss
                        # We keep track of the loss at each epoch
                        if args.with_tracking:
                            total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break
        model.eval()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
        }
        debug_flag = 1
        accelerator.print("eval" + "!" * 1000)
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            with torch.no_grad():

                if args.encoder_prompt == "kmeans":
                    outputs = model(**batch)
                    batch = batch.to('cpu')
                    inputs = cluster_prompt(batch, outputs)
                    del outputs
                    inputs = [prefix + inp for inp in inputs]
                    model_inputs = [
                        tokenizer(input_, max_length=args.max_source_length, padding=padding, truncation=True) for input_ in inputs]
                    features = data_collator(model_inputs)
                    for key in ['input_ids', 'attention_mask']:
                        batch[key] = features[key]
                    batch = batch.to(accelerator.device)

                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                batch = batch.to('cpu')
                del batch

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                if debug_flag:
                    accelerator.print(f"==============epoch {epoch} raw_decoded_preds==============", file=debug_file)
                    accelerator.print(decoded_preds, file=debug_file)
                    accelerator.print(f"==============epoch {epoch} raw_decoded_labels==============", file=debug_file)
                    accelerator.print(decoded_labels, file=debug_file)
                    debug_file.flush()
                if args.decoder_prompt == "svo":
                    def mask_svo(t):
                        pos = t.find(prompt_sep_token)
                        return t[(pos + len(prompt_sep_token)) if pos != -1 else 0:].strip()

                    decoded_preds = [mask_svo(x) for x in decoded_preds]
                    decoded_labels = [mask_svo(x) for x in decoded_labels]
                    if debug_flag:
                        accelerator.print(f"==============epoch {epoch} decoded_preds==============", file=debug_file)
                        accelerator.print(decoded_preds, file=debug_file)
                        accelerator.print(f"==============epoch {epoch} decoded_labels==============", file=debug_file)
                        accelerator.print(decoded_labels, file=debug_file)
                        debug_file.flush()
                if debug_flag:
                    debug_flag -= 1

                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        if args.target_domain_dataset_name is not None:
            for step, batch in enumerate(target_domain_eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    labels = batch["labels"]
                    if not args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                    generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                    generated_tokens = generated_tokens.cpu().numpy()
                    labels = labels.cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                    metric.add_batch(
                        predictions=decoded_preds,
                        references=decoded_labels,
                    )
            target_domain_result = metric.compute(use_stemmer=True)
            target_domain_result = {k: round(v * 100, 4) for k, v in target_domain_result.items()}
            for k, v in target_domain_result.items():
                result[f"target_domain_{k}"] = v

        logger.info(result)

        if args.with_tracking:
            if not args.test:
                result["train_loss"] = total_loss.item() / len(train_dataloader)  # 生成式loss
            result["epoch"] = epoch
            result["step"] = completed_steps
            if args.model_type == "BartForConditionalGenerationWithRouge1":
                result["total_ext_rouge1_loss"] = total_ext_rouge1_loss.item() / len(train_dataloader)  # rouge 1 loss
                result["loss_sum"] = result["train_loss"] + result["total_ext_rouge1_loss"]  # 生成式 + rouge 1 loss
            if args.model_type == "BartForConditionalGenerationWithRouge1Rouge2":
                result["total_ext_rouge1_loss"] = total_ext_rouge1_loss.item() / len(train_dataloader)  # rouge 1 loss
                result["total_ext_rouge2_loss"] = total_ext_rouge2_loss.item() / len(train_dataloader)  # rouge 2 loss
                result["loss_sum"] = result["train_loss"] + result["total_ext_rouge1_loss"] + result["total_ext_rouge2_loss"]  # 生成式 + rouge 1 loss + rouge 2 loss
            accelerator.log(result, step=completed_steps)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            all_results = {f"eval_{k}": v for k, v in result.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)


if __name__ == "__main__":
    main()