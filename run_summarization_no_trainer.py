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
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset, DatasetDict
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
from transformers.models.bart.modeling_bart import BartDecoder
from transformers.utils import check_min_version, get_full_repo_name, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from bart_with_ext import BartForConditionalGenerationWithMultiCLS

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
        "--cached_dataset_path",
        type=str,
        default=None,
        help="The path datasets cached.",
    )
    parser.add_argument(
        "--use_cached_dataset",
        type=bool,
        default=False,
        help="If use cached dataset.",
    )
    parser.add_argument(
        "--filtered_dataset_path",
        type=str,
        default=None,
        help="The path datasets cached.",
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
    parser.add_argument(
        "--sep_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--cls_step1_threhold",
        type=float,
        default=0.8,
        help="Cls step1 threhold.",
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
        "--generated_dataset_path",
        type=str,
        default=None,
        help="generated dataset path.",
    )
    parser.add_argument(
        "--generate_mode",
        action="store_true",
        help="Only generate dataset",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only do test",
    )
    parser.add_argument(
        "--test_epoch",
        type=str,
        default=None,
        help="test epoch dir.",
    )
    parser.add_argument(
        "--debug",
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
    from DataCollatorForBartForConditionalGenerationWithRouge import DataCollatorForBartForConditionalGenerationWithRouge, DataCollatorForMaskRouge, DataCollatorMultiCLS

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    accelerator.print(json.dumps(args.__dict__, indent=4, ensure_ascii=False, sort_keys=True))
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
        if args.test:
            config = AutoConfig.from_pretrained(args.output_dir)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.encoder_prompt == "multi_cls_step1":
        cls_config = AutoConfig.from_pretrained('microsoft/deberta-v3-base', trust_remote_code=True)
        for attr_name in dir(cls_config):
            if hasattr(config, attr_name):
                continue
            attr_value = getattr(cls_config, attr_name)
            accelerator.print(f"set attr '{attr_name}': '{attr_value}' to config")
            setattr(config, attr_name, attr_value)
            setattr(config, 'pooler_hidden_size', getattr(config, 'd_model'))

    accelerator.print(config)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    cls_label_column = 'cls_labels'
    if args.model_name_or_path:  # todo: test 的时候加载训练过的模型
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
        elif args.encoder_prompt == "multi_cls_step1":
            if args.test or args.generate_mode:
                model = BartForConditionalGenerationWithMultiCLS.from_pretrained(
                    f"{args.output_dir}/{args.test_epoch}/pytorch_model.bin",
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                )
                logger.info("Loading model from {args.output_dir}/{args.test_epoch}")
            else:
                model = BartForConditionalGenerationWithMultiCLS.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                )
            model.cls_label_column = cls_label_column
            del model.model.decoder
            del model.lm_head
        else:
            if args.test:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    f"{args.output_dir}/{args.test_epoch}/pytorch_model.bin",
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                )

        if args.encoder_prompt == "kmeans" or args.encoder_prompt == "contrastive_kmeans":
            encoder = model.get_encoder()

        if args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
            sep_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
            sep_model.model.encoder = model.get_encoder()
            sep_model.get_decoder().set_input_embeddings(model.get_encoder().embed_tokens)
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

    punctuations = [',', '.', ';', '"', "'", '?', '!', ':']
    def clean(example):
        example[text_column] = unicodedata.normalize('NFKC', example[text_column])
        example[text_column] = ' '.join([x for x in example[text_column].split(' ') if x != ''])

        from nltk.tokenize import sent_tokenize
        highlights_sents = sent_tokenize(example[summary_column])
        if len(highlights_sents) == 1:
            example[summary_column] = example[summary_column].replace("\n", ". \n")
            highlights_sents = sent_tokenize(example[summary_column])
        if len(highlights_sents) == 1:
            example[summary_column] = example[summary_column].replace("  ", ". \n")

        example[summary_column] = unicodedata.normalize('NFKC', example[summary_column])
        example[summary_column] = ' '.join([x for x in example[summary_column].split(' ') if x != ''])

        for punctuation in punctuations:
            example[text_column] = example[text_column].replace(f'{punctuation} ', f'{punctuation}').replace(f'{punctuation}', f'{punctuation} ')
        example[text_column] = ' ' + example[text_column].lstrip()
        example[text_column] = prefix + example[text_column]

        for punctuation in punctuations:
            example[summary_column] = example[summary_column].replace(f'{punctuation} ', f'{punctuation}').replace(f'{punctuation}', f'{punctuation} ')
        example[summary_column] = example[summary_column].replace("\n", "")  # 很重要，能影响三个点，摘要里不能加\n
        example[summary_column] = ' ' + example[summary_column].lstrip()

        return example

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    def dataset_filter(example):
        from nltk.tokenize import sent_tokenize
        tmp_ids = tokenizer(example[text_column], max_length=args.max_source_length, truncation=True).input_ids
        article = tokenizer.decode(tmp_ids)
        article_sents = sent_tokenize(article)
        highlights_sents = sent_tokenize(example[summary_column])
        # highlights_sents = example[summary_column].split('\n')
        if len(article_sents) <= 3:
            return False

        if any([len([t for t in x.split(' ') if t != '']) < 4 for x in example[summary_column].split('\n')]):
            return False

        flag = 0
        for i, article_sent in enumerate(article_sents):
            rouges = []
            for j, highlights_sent in enumerate(highlights_sents):
                score = scorer.score(highlights_sent, article_sent)["rouge2"].fmeasure
                rouges.append((j + 1, score))
            rouges = sorted(rouges, key=lambda x: x[1], reverse=True)
            if rouges[0][1] > 0:
                flag = 1
        return flag > 0

    accelerator.print(raw_datasets)
    if not args.use_cached_dataset:
        try:
            raw_datasets = load_from_disk(args.filtered_dataset_path)
        except:
            with accelerator.main_process_first():
                raw_datasets = raw_datasets.map(clean, num_proc=args.preprocessing_num_workers)
                raw_datasets = raw_datasets.filter(dataset_filter, num_proc=args.preprocessing_num_workers)
            raw_datasets.save_to_disk(args.filtered_dataset_path)
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
        article_sents = sent_tokenize(example[text_column])
        highlights_sents = sent_tokenize(example[summary_column])

        for i, article_sent in enumerate(article_sents):
            rouges = []
            for j, highlights_sent in enumerate(highlights_sents):
                score = scorer.score(highlights_sent, article_sent)["rouge2"].fmeasure
                rouges.append((j + 1, score))
            rouges = sorted(rouges, key=lambda x: x[1], reverse=True)
            if rouges[0][1] > 0:
                article_sents[i] = f'[{rouges[0][0]}] {article_sents[i]} [/{rouges[0][0]}]'
        example[text_column] = ' '.join(article_sents)
        return example

    def get_lightweight_rouge_prompt_article(example):
        from nltk.tokenize import sent_tokenize
        article_sents = sent_tokenize(example[text_column])

        point_flag = False
        for i, article_sent in enumerate(article_sents):
            score = scorer.score(example[summary_column], article_sent)["rouge2"].fmeasure
            if score > 0:
                if point_flag:
                    article_sents[i] = f'{article_sents[i]} ]'
                    article_sents[i - 1] = article_sents[i - 1][:-2]
                else:
                    article_sents[i] = f'[ {article_sents[i]} ]'
                point_flag = True
            else:
                point_flag = False
        example[text_column] = ' '.join(article_sents)
        return example

    mask_text_column = 'mask_text'
    mask_label_column = 'mask_label'
    def get_mask_rouge_prompt_article(example):
        from nltk.tokenize import sent_tokenize
        article_sents = sent_tokenize(example[text_column])
        highlights_sents = sent_tokenize(example[summary_column])
        mask_label_sents = []

        for i, article_sent in enumerate(article_sents):
            rouges = []
            for j, highlights_sent in enumerate(highlights_sents):
                score = scorer.score(highlights_sent, article_sent)["rouge2"].fmeasure
                rouges.append((j + 1, score))
            rouges = sorted(rouges, key=lambda x: x[1], reverse=True)
            if rouges[0][1] > 0:
                mask_label_sent = f'[{rouges[0][0]}] {article_sents[i]} [/{rouges[0][0]}]'
            else:
                mask_label_sent = article_sents[i]
            mask_label_sents.append(mask_label_sent)
            article_sents[i] = f'{tokenizer.mask_token}{article_sents[i]}'
        example[text_column] = ' '.join(article_sents) + f'{tokenizer.mask_token}'
        example[mask_label_column] = ' '.join(mask_label_sents)
        return example

    def get_mask_rouge_prompt_article_lightweight(example):
        from nltk.tokenize import sent_tokenize
        article_sents = sent_tokenize(example[text_column])
        mask_label_sents = []

        point_flag = False
        for i, article_sent in enumerate(article_sents):
            score = scorer.score(example[summary_column], article_sent)["rouge2"].fmeasure
            if score > 0:
                if point_flag:
                    mask_label_sent = f'{article_sents[i]} ]'
                    mask_label_sents[-1] = mask_label_sents[-1][:-2]
                else:
                    mask_label_sent = f'[ {article_sents[i]} ]'
                point_flag = True
            else:
                point_flag = False
                mask_label_sent = article_sents[i]
            mask_label_sents.append(mask_label_sent)
            article_sents[i] = f'{tokenizer.mask_token}{article_sents[i]}'
        example[mask_text_column] = ' '.join(article_sents) + f'{tokenizer.mask_token}'
        example[mask_label_column] = ' '.join(mask_label_sents)
        return example

    def get_mask_multi_cls(example):
        from nltk.tokenize import sent_tokenize
        article_sents = sent_tokenize(example[text_column])
        cls_labels = []

        for i, article_sent in enumerate(article_sents):
            if len([x for x in article_sent.split(' ') if x.strip()]) > 2:
                score = scorer.score(example[summary_column], article_sent)["rouge2"].fmeasure
                if score > 0:
                    cls_labels.append(1)
                else:
                    cls_labels.append(0)
                article_sents[i] = f'{tokenizer.mask_token}{article_sents[i]}'
        example[text_column] = ' '.join(article_sents)
        example[cls_label_column] = cls_labels
        return example

    def get_mask_rouge_prompt_article_step1(example):
        from nltk.tokenize import sent_tokenize
        article_sents = sent_tokenize(example[text_column])
        highlights_sents = sent_tokenize(example[summary_column])
        mask_label_sents = []

        for i, article_sent in enumerate(article_sents):
            rouges = []
            for j, highlights_sent in enumerate(highlights_sents):
                score = scorer.score(highlights_sent, article_sent)["rouge2"].fmeasure
                rouges.append((j + 1, score))
            rouges = sorted(rouges, key=lambda x: x[1], reverse=True)
            if rouges[0][1] > 0:
                mask_label_sent = f'[{rouges[0][0]}] {article_sents[i]} [/{rouges[0][0]}]'
            else:
                mask_label_sent = article_sents[i]
            mask_label_sents.append(mask_label_sent)
            article_sents[i] = f'{tokenizer.mask_token}{article_sents[i]}'
        example[text_column] = ' '.join(article_sents) + f'{tokenizer.mask_token}'
        example[summary_column] = ' '.join(mask_label_sents)
        return example

    max_mask_target_length = args.max_source_length
    mask_input_ids_name = "mask_input_ids"
    mask_attention_mask_name = "mask_attention_mask"
    mask_labels_ids_name = "mask_labels"
    def preprocess_function(examples):
        inputs = examples[text_column]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        targets = examples[summary_column]
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]

        if args.encoder_prompt == "multi_cls_step1" and not args.generate_mode:
            cls_labels = examples[cls_label_column]
            model_inputs[cls_label_column] = []
            for i, input_ids in enumerate(model_inputs['input_ids']):
                cls_count = 0
                cls_label = []
                for token in input_ids:
                    if token == tokenizer.mask_token_id:
                        cls_label.append(cls_labels[i][cls_count])
                        cls_count += 1
                    else:
                        cls_label.append(-100)
                model_inputs[cls_label_column].append(cls_label)

        if args.encoder_prompt == "mask_rouge" or args.encoder_prompt == "mask_rouge_lightweight" or args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
            mask_inputs = examples[mask_text_column]
            mask_model_inputs = tokenizer(mask_inputs, max_length=args.max_source_length, padding=padding, truncation=True)

            mask_targets = examples[mask_label_column]
            mask_labels = tokenizer(text_target=mask_targets, max_length=max_mask_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                mask_labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in mask_labels["input_ids"]
                ]
            model_inputs[mask_input_ids_name] = mask_model_inputs["input_ids"]
            model_inputs[mask_attention_mask_name] = mask_model_inputs["attention_mask"]
            model_inputs[mask_labels_ids_name] = mask_labels["input_ids"]

        if args.model_type == "BartForConditionalGenerationWithRouge1":
            model_inputs["ext_rouge1_labels"] = get_batch_sailent_scores_rouge1(model_inputs["input_ids"], model_inputs["labels"])
        elif args.model_type == "BartForConditionalGenerationWithRouge1Rouge2":
            model_inputs["ext_rouge1_labels"] = get_batch_sailent_scores_rouge1(model_inputs["input_ids"], model_inputs["labels"])
            model_inputs["ext_rouge2_labels"] = get_batch_sailent_scores_rouge2(model_inputs["input_ids"], model_inputs["labels"])
        return model_inputs

    def split_sent_embedding_from_outputs(input_ids, encoder_last_hidden_state, first_use_n=False):
        from nltk.tokenize import sent_tokenize
        input_ids[input_ids == -100] = tokenizer.pad_token_id
        article = tokenizer.decode(input_ids, skip_special_tokens=True)
        for punctuation in punctuations:
            article = article.replace(f'{punctuation} ', f'{punctuation}').replace(f'{punctuation}', f'{punctuation} ')
        if first_use_n:
            sents = article.split('\n')
        else:
            sents = sent_tokenize(article)
        embeddings = []
        start_position = 1
        for sent_id, sent in enumerate(sents):
            if sent_id != 0:
                sent = ' ' + sent
            sent_len = len(tokenizer(sent).input_ids) - 2
            sentence_embedding = encoder_last_hidden_state[start_position: start_position + sent_len]
            embeddings.append(torch.mean(sentence_embedding, dim=0))
            start_position += sent_len
        embeddings = torch.stack(embeddings)
        return article, sents, embeddings

    def cluster(np_emb, sents, input_ids, encoder_last_hidden_state):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3)
        try:
            kmeans = kmeans.fit(np_emb)
        except:
            traceback.print_exc()
            start_position = 1
            for sent_id, sent in enumerate(sents):
                if sent_id != 0:
                    sent = ' ' + sent
                sent_len = len(tokenizer(sent).input_ids) - 2
                sentence_embedding = encoder_last_hidden_state[start_position: start_position + sent_len]
                print('--------sentence_embedding--------')
                print(sentence_embedding)
                print(sentence_embedding.size())
                print(sent)
                print(tokenizer(sent).input_ids)
                print(sent_len)
                print(input_ids[start_position: start_position + sent_len])
                print(tokenizer.decode(input_ids[start_position: start_position + sent_len], skip_special_tokens=False))
                start_position += sent_len
            print('--------encoder_last_hidden_state--------')
            print(input_ids)
            print(np_emb)
            print(encoder_last_hidden_state)
            if args.debug:
                raise ValueError("模型输出不符合预期")
            return ' '.join(sents)

        # km_labels = km.labels_
        # print(km_labels)

        for n_cluster_index in range(kmeans.n_clusters):
            distances = kmeans.transform(np_emb)[:, n_cluster_index]
            # print(len([closest_i for closest_i in np.argsort(distances) if kmeans.labels_[closest_i] == n_cluster_index]))
            # closest = [closest_i for closest_i in np.argsort(distances) if kmeans.labels_[closest_i] == n_cluster_index][:3]
            closest = np.argsort(distances)[:3]
            for center_sent_index in closest:
                sents[center_sent_index] = f'[{n_cluster_index + 1}] {sents[center_sent_index]} [/{n_cluster_index + 1}]'
        article = ' '.join(sents)
        return article

    def cluster_prompt(batch, encoder_last_hidden_states):
        articles = []
        for article_index, encoder_last_hidden_state in enumerate(encoder_last_hidden_states):
            input_ids = batch.input_ids[article_index]
            article, sents, embeddings = split_sent_embedding_from_outputs(input_ids, encoder_last_hidden_state)
            np_emb = embeddings.cpu().detach().numpy()
            article = cluster(np_emb, sents, input_ids, encoder_last_hidden_state)
            articles.append(article)
        return articles

    def rouge_related_contrastive_loss(article_sents, article_embeddings, highlights_sents, highlight_embeddings):
        from ContrastiveLoss import ContrastiveLoss
        contrastive_loss_fn = ContrastiveLoss()

        left_examples = []
        right_examples = []
        contrastive_labels = []

        for i, article_sent in enumerate(article_sents):
            rouges = []
            for j, highlights_sent in enumerate(highlights_sents):
                score = scorer.score(highlights_sent, article_sent)["rouge2"].fmeasure
                rouges.append((j, score))
            rouges = sorted(rouges, key=lambda x: x[1], reverse=True)
            if rouges[0][1] > 0:  # 正样本对
                left_examples.append(article_embeddings[i])
                right_examples.append(highlight_embeddings[rouges[0][0]])
                contrastive_labels.append(1)

        # 负样本对
        for i in range(highlight_embeddings.size()[0]):
            for j in range(i + 1, highlight_embeddings.size()[0]):
                left_examples.append(highlight_embeddings[i])
                right_examples.append(highlight_embeddings[j])
                contrastive_labels.append(0)

        try:
            left_examples = torch.stack(left_examples)
            right_examples = torch.stack(right_examples)
            contrastive_labels = torch.tensor(contrastive_labels).to(accelerator.device)
            # contrastive_labels = contrastive_labels.to(accelerator.device)
            contrastive_loss = contrastive_loss_fn(left_examples, right_examples, contrastive_labels)
        except:
            traceback.print_exc()
            print(article_sents)
            print(article_embeddings)
            print(highlights_sents)
            print(highlight_embeddings)
            if args.debug:
                raise ValueError("对比学习不符合预期")
            return torch.tensor(0.0).to(accelerator.device)
        return contrastive_loss

    def contrastive_cluster_prompt(batch, encoder_last_hidden_states, label_encoder_last_hidden_states):
        articles = []
        contrastive_losses = []
        for article_index, encoder_last_hidden_state in enumerate(encoder_last_hidden_states):
            input_ids = batch.input_ids[article_index]
            article, sents, embeddings = split_sent_embedding_from_outputs(input_ids, encoder_last_hidden_state)

            label_input_ids = batch.labels[article_index]
            label_encoder_last_hidden_state = label_encoder_last_hidden_states[article_index]
            highlight, highlights_sents, highlight_embeddings = split_sent_embedding_from_outputs(label_input_ids, label_encoder_last_hidden_state, first_use_n=False)

            contrastive_loss = rouge_related_contrastive_loss(sents, embeddings, highlights_sents, highlight_embeddings)
            contrastive_losses.append(contrastive_loss)

            np_emb = embeddings.cpu().detach().numpy()
            article = cluster(np_emb, sents, input_ids, encoder_last_hidden_state)
            articles.append(article)

        contrastive_loss_sum = torch.sum(torch.stack(contrastive_losses))
        return articles, contrastive_loss_sum



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

    if not args.use_cached_dataset:
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
                    desc="Running rouge prompt on dataset",
                )

            if args.encoder_prompt == "lightweight_rouge":
                raw_datasets = raw_datasets.map(
                    get_lightweight_rouge_prompt_article,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc="Running rouge prompt on dataset",
                )

            if args.encoder_prompt == "mask_rouge":
                raw_datasets = raw_datasets.map(
                    get_mask_rouge_prompt_article,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc="Running mask rouge prompt on dataset",
                )

            if args.encoder_prompt == "mask_rouge_lightweight" or args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
                raw_datasets = raw_datasets.map(
                    get_mask_rouge_prompt_article_lightweight,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc="Running mask rouge prompt lightweight on dataset",
                )

            if args.encoder_prompt == "multi_cls_step1":
                raw_datasets = raw_datasets.map(
                    get_mask_multi_cls,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc="Running cls lightweight on dataset",
                )

            if args.encoder_prompt == "mask_rouge_step1":
                raw_datasets = raw_datasets.map(
                    get_mask_rouge_prompt_article_step1,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc="Running mask rouge prompt step1 on dataset",
                )

        with accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )
            if args.encoder_prompt == "mask_rouge" or args.encoder_prompt == "mask_rouge_lightweight" or args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
                processed_datasets = processed_datasets.remove_columns([mask_text_column, mask_label_column])
            if args.target_domain_dataset_name is not None:
                processed_target_domain_datasets = target_domain_raw_datasets.map(
                    preprocess_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=True,
                    desc="Running tokenizer on target domain dataset",
                )
        processed_datasets.save_to_disk(args.cached_dataset_path)

    if args.use_cached_dataset:
        processed_datasets = load_from_disk(args.cached_dataset_path)

    if args.encoder_prompt == "multi_cls_step1":
        if args.generate_mode:
            train_generate_list = []
            test_generate_list = []
        else:
            processed_datasets = processed_datasets.remove_columns(['labels'])

    if args.encoder_prompt == "multi_cls_step2":
        processed_datasets = processed_datasets.remove_columns([text_column, summary_column])

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
    elif args.encoder_prompt == "mask_rouge" or args.encoder_prompt == "mask_rouge_lightweight" or args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
        data_collator = DataCollatorForMaskRouge(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
        data_collator.mask_labels_ids_name = mask_labels_ids_name
        second_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
    elif args.encoder_prompt == "multi_cls_step1":
        data_collator = DataCollatorMultiCLS(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
        data_collator.cls_label_column = cls_label_column
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
    if args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
        sep_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in sep_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in sep_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        sep_optimizer = torch.optim.AdamW(sep_optimizer_grouped_parameters, lr=args.sep_learning_rate)

        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": args.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        #     {
        #         "params": [p for n, p in sep_model.get_decoder().named_parameters() if not any(nd in n for nd in no_decay) and n not in ['embed_tokens.weight']],
        #         "weight_decay": args.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in sep_model.get_decoder().named_parameters() if any(nd in n for nd in no_decay) and n not in ['embed_tokens.weight']],
        #         "weight_decay": 0.0,
        #     }
        # ]
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        #

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
    if args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
        sep_lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=sep_optimizer,
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
    elif args.encoder_prompt == "kmeans" or args.encoder_prompt == "contrastive_kmeans":
        model, encoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, encoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    elif args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
        model, sep_model, optimizer, sep_optimizer, train_dataloader, eval_dataloader, lr_scheduler, sep_lr_scheduler = accelerator.prepare(
            model, sep_model, optimizer, sep_optimizer, train_dataloader, eval_dataloader, lr_scheduler, sep_lr_scheduler
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
    if args.test or args.generate_mode:
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
    if args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
        sep_metric = evaluate.load("rouge")
    if args.encoder_prompt == "multi_cls_step1":
        acc_metric = evaluate.load("accuracy")
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
            samples_seen = 0
            total_loss = 0
            total_cls_loss = 0
            total_contrastive_loss = 0
            total_mask_loss = 0
            total_ext_rouge1_loss = 0
            total_ext_rouge2_loss = 0
            for step, batch in enumerate(train_dataloader):
                # accelerator.print(batch.keys()) # ['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                if completed_steps <= args.resume_step:
                    if completed_steps == args.resume_step:
                        logger.info("***** Resume Done *****")
                    else:
                        progress_bar.update(1)
                        completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    if args.encoder_prompt == "kmeans":
                        encoder_last_hidden_states = encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state.to('cpu')
                        batch = batch.to('cpu')
                        # accelerator.print(outputs.keys()) # dict_keys(['loss', 'logits', 'encoder_last_hidden_state'])
                        inputs = cluster_prompt(batch, encoder_last_hidden_states)
                        del encoder_last_hidden_states

                        inputs = [prefix + inp for inp in inputs]
                        model_inputs = [tokenizer(input_, max_length=args.max_source_length, padding=padding, truncation=True) for input_ in inputs]
                        features = data_collator(model_inputs)
                        for key in ['input_ids', 'attention_mask']:
                            batch[key] = features[key]
                        batch = batch.to(accelerator.device)
                        outputs = model(**batch)
                        batch = batch.to('cpu')
                        del batch
                        loss = outputs.loss
                        del outputs
                        # We keep track of the loss at each epoch
                        if args.with_tracking:
                            log_gen_loss = loss.cpu().detach().float()
                            accelerator.log({"batch_gen_loss": log_gen_loss}, step=completed_steps)
                            total_loss += log_gen_loss

                    elif args.encoder_prompt == "contrastive_kmeans":
                        encoder_last_hidden_states = encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
                        batch = batch.to('cpu')
                        # accelerator.print(outputs.keys()) # dict_keys(['loss', 'logits', 'encoder_last_hidden_state'])

                        labels_without_negative_values = batch['labels'].clone()
                        labels_without_negative_values[labels_without_negative_values == -100] = tokenizer.pad_token_id
                        highlights = tokenizer.batch_decode(labels_without_negative_values, skip_special_tokens=True)
                        model_inputs = [
                            tokenizer(input_, max_length=args.max_target_length, padding=padding, truncation=True) for
                            input_ in highlights]
                        features = data_collator(model_inputs)
                        features = features.to(accelerator.device)
                        label_encoder_last_hidden_states = encoder(input_ids=features['input_ids'], attention_mask=features['attention_mask']).last_hidden_state
                        features = features.to('cpu')
                        del features
                        # accelerator.print(outputs.keys()) # dict_keys(['logits', 'past_key_values', 'encoder_last_hidden_state'])

                        inputs, contrastive_loss = contrastive_cluster_prompt(batch, encoder_last_hidden_states, label_encoder_last_hidden_states)

                        # with torch.cuda.device(accelerator.device):cls_forward
                        #     torch.cuda.empty_cache()

                        inputs = [prefix + inp for inp in inputs]
                        model_inputs = [
                            tokenizer(input_, max_length=args.max_source_length, padding=padding, truncation=True) for
                            input_ in inputs]
                        features = data_collator(model_inputs)
                        for key in ['input_ids', 'attention_mask']:
                            batch[key] = features[key]
                        batch = batch.to(accelerator.device)
                        outputs = model(**batch)
                        # accelerator.print(outputs.keys())
                        batch = batch.to('cpu')
                        del batch
                        # contrastive_loss = contrastive_loss.to(accelerator.device)
                        if torch.isnan(contrastive_loss).any().item():  # 有 nan 值
                            loss = outputs.loss
                        else:
                            loss = outputs.loss + contrastive_loss
                        # We keep track of the loss at each epoch
                        if args.debug:
                            print(f'gen loss: {outputs.loss.cpu().detach().float()}')
                            print(f'contrastive loss: {contrastive_loss.cpu().detach().float()}')
                        if args.with_tracking:
                            log_gen_loss = outputs.loss.cpu().detach().float()
                            log_contrastive_loss = contrastive_loss.cpu().detach().float()
                            accelerator.log({"batch_gen_loss": log_gen_loss, "batch_contrastive_loss": log_contrastive_loss}, step=completed_steps)
                            total_loss += log_gen_loss
                            total_contrastive_loss += log_contrastive_loss

                    elif args.encoder_prompt == "multi_cls_step1":  # todo: label smooth
                        if args.generate_mode:
                            with torch.no_grad():
                                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                                labels=batch[cls_label_column], cls_mode=True)
                                # predictions = outputs.logits.argmax(dim=-1)
                                predictions = (outputs.logits.softmax(dim=-1).view(-1, 2)[:, 1] >= args.cls_step1_threhold).long().view(
                                    batch['input_ids'].shape[0], batch['input_ids'].shape[1], 1)
                                references = batch[cls_label_column]
                                if args.generate_mode:
                                    input_idss, labels = accelerator.gather((batch['input_ids'], batch['labels']))
                                predictions, references = accelerator.gather((predictions, references))
                                # If we are in a multiprocess environment, the last batch has duplicates
                                if accelerator.num_processes > 1:
                                    if step == len(train_dataloader) - 1:
                                        predictions = predictions[: len(train_dataloader.dataset) - samples_seen]
                                        references = references[: len(train_dataloader.dataset) - samples_seen]
                                    else:
                                        samples_seen += references.shape[0]
                                predictions = [[int(x) for j, x in enumerate(prediction) if references[i][j] != -100]
                                               for i, prediction in enumerate(predictions)]
                                references = [[int(x) for x in reference if x != -100] for reference in references]

                                if args.generate_mode:
                                    for example_index in range(len(predictions)):
                                        input_ids = input_idss[example_index]
                                        label = labels[example_index]
                                        prediction = predictions[example_index]
                                        # cls_label = references[example_index]
                                        input_ids = [x for x in input_ids if x != tokenizer.pad_token_id]
                                        # prediction = [x for i, x in enumerate(prediction) if cls_label[i] != -100]
                                        label = [x for x in label if x != -100]
                                        sents = tokenizer.decode(input_ids[1:-1]).split(tokenizer.mask_token)
                                        highlight = tokenizer.decode(label, skip_special_tokens=True)
                                        article_with_importance = [sents[0]]
                                        for i, sent in enumerate(sents[1:]):
                                            if prediction[i] == 1:
                                                sent = f'{tokenizer.mask_token} {sent}'
                                            article_with_importance.append(sent)
                                        article = ' '.join(article_with_importance)
                                        train_generate_list.append({'article': article, 'highlights': highlight})
                        else:
                            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch[cls_label_column], cls_mode=True)
                            loss = outputs.loss
                            if args.with_tracking:
                                log_cls_loss = outputs.loss.cpu().detach().float()
                                accelerator.log({"batch_cls_loss": log_cls_loss}, step=completed_steps)
                                total_cls_loss += log_cls_loss
                    elif args.encoder_prompt == "mask_rouge" or args.encoder_prompt == "mask_rouge_lightweight":
                        # labels = batch['labels']
                        # batch['labels'] = batch[mask_labels_ids_name]
                        # del batch[mask_labels_ids_name]
                        #
                        # decoder_input_ids = batch['labels_decoder_input_ids']
                        # batch['decoder_input_ids'] = batch[f"{mask_labels_ids_name}_decoder_input_ids"]
                        # del batch[f"{mask_labels_ids_name}_decoder_input_ids"]
                        # del batch['labels_decoder_input_ids']
                        #
                        # outputs = model(**batch)
                        # mask_loss = outputs.loss
                        #
                        #
                        # generated_tokens = accelerator.unwrap_model(model).generate(
                        #     batch["input_ids"],
                        #     attention_mask=batch["attention_mask"],
                        #     **mask_gen_kwargs,
                        # )
                        # generated_tokens = generated_tokens.cpu().numpy()
                        # if isinstance(generated_tokens, tuple):
                        #     generated_tokens = generated_tokens[0]
                        # decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        # inputs = [prefix + inp for inp in decoded_preds]
                        # model_inputs = [
                        #     tokenizer(input_, max_length=args.max_source_length, padding=padding, truncation=True) for
                        #     input_ in inputs]
                        # # todo decoder input ids
                        # batch = second_data_collator(model_inputs)
                        # batch['labels'] = labels
                        # batch['decoder_input_ids'] = decoder_input_ids
                        # batch = batch.to(accelerator.device)
                        # outputs = model(**batch)
                        # loss = outputs.loss + mask_loss
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                        labels=batch['labels'], decoder_input_ids=batch['labels_decoder_input_ids'])
                        gen_loss = outputs.loss

                        outputs = model(input_ids=batch[mask_input_ids_name], attention_mask=batch[mask_attention_mask_name],
                                        labels=batch[mask_labels_ids_name], decoder_input_ids=batch[f"{mask_labels_ids_name}_decoder_input_ids"])
                        mask_loss = outputs.loss

                        loss = gen_loss + mask_loss

                        if args.debug:
                            print(f'gen loss: {gen_loss.cpu().detach().float()}')
                            print(f'mask loss: {mask_loss.cpu().detach().float()}')
                        if args.with_tracking:
                            log_gen_loss = gen_loss.cpu().detach().float()
                            log_mask_loss = mask_loss.cpu().detach().float()
                            accelerator.log({"batch_gen_loss": log_gen_loss, "batch_mask_loss": log_mask_loss}, step=completed_steps)
                            total_loss += log_gen_loss
                            total_mask_loss += log_mask_loss

                    elif args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
                        outputs = sep_model(input_ids=batch[mask_input_ids_name],
                                            attention_mask=batch[mask_attention_mask_name],
                                            labels=batch[mask_labels_ids_name],
                                            decoder_input_ids=batch[f"{mask_labels_ids_name}_decoder_input_ids"])
                        mask_loss = outputs.loss
                        accelerator.backward(mask_loss)
                        sep_optimizer.step()
                        sep_lr_scheduler.step()
                        sep_optimizer.zero_grad()

                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                        labels=batch['labels'], decoder_input_ids=batch['labels_decoder_input_ids'])
                        gen_loss = outputs.loss

                        loss = gen_loss

                        if args.debug:
                            print(f'gen loss: {gen_loss.cpu().detach().float()}')
                            print(f'mask loss: {mask_loss.cpu().detach().float()}')
                        if args.with_tracking:
                            log_gen_loss = gen_loss.cpu().detach().float()
                            log_mask_loss = mask_loss.cpu().detach().float()
                            accelerator.log({"batch_gen_loss": log_gen_loss, "batch_mask_loss": log_mask_loss, "mask_learning_rate": sep_optimizer.param_groups[0]['lr']},
                                            step=completed_steps)
                            total_loss += log_gen_loss
                            total_mask_loss += log_mask_loss

                    elif args.model_type == "BartForConditionalGenerationWithRouge1":
                        outputs = model(**batch)
                        abs_loss = outputs.loss
                        masked_ext_rouge1_loss = outputs.masked_ext_rouge1_loss
                        # We keep track of the loss at each epoch
                        if args.with_tracking:
                            total_loss += abs_loss.detach().float()
                            total_ext_rouge1_loss += masked_ext_rouge1_loss.detach().float()
                        loss = abs_loss + masked_ext_rouge1_loss
                    elif args.model_type == "BartForConditionalGenerationWithRouge1Rouge2":
                        outputs = model(**batch)
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
                        outputs = model(**batch)
                        loss = outputs.loss
                        # We keep track of the loss at each epoch
                        if args.with_tracking:
                            log_gen_loss = loss.cpu().detach().float()
                            accelerator.log({"batch_gen_loss": log_gen_loss}, step=completed_steps)
                            total_loss += loss.detach().float()
                    if args.with_tracking:
                        accelerator.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=completed_steps)
                    if not args.generate_mode:
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int) and not args.test:
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

        model.eval()
        samples_seen = 0
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
            # "early_stopping": args.early_stopping,
            # "length_penalty": args.length_penalty,
            # "min_length": args.min_length,
            # "no_repeat_ngram_size": args.no_repeat_ngram_size
        }
        debug_flag = 1
        accelerator.print("eval" + "!" * 1000)
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            with torch.no_grad():

                if args.encoder_prompt == "kmeans" or args.encoder_prompt == "contrastive_kmeans":
                    outputs = model(**batch)
                    batch = batch.to('cpu')
                    outputs.encoder_last_hidden_state = outputs.encoder_last_hidden_state.to('cpu')
                    encoder_last_hidden_states = outputs.encoder_last_hidden_state
                    inputs = cluster_prompt(batch, encoder_last_hidden_states)
                    del outputs.past_key_values
                    del outputs.logits
                    del outputs
                    inputs = [prefix + inp for inp in inputs]
                    model_inputs = [
                        tokenizer(input_, max_length=args.max_source_length, padding=padding, truncation=True) for input_ in inputs]
                    features = data_collator(model_inputs)
                    for key in ['input_ids', 'attention_mask']:
                        batch[key] = features[key]
                    batch = batch.to(accelerator.device)

                if args.encoder_prompt == "multi_cls_step1":
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch[cls_label_column], cls_mode=True)
                    # predictions = outputs.logits.argmax(dim=-1)
                    predictions = (outputs.logits.softmax(dim=-1).view(-1, 2)[:, 1] >= args.cls_step1_threhold).long().view(batch['input_ids'].shape[0], batch['input_ids'].shape[1], 1)
                    references = batch[cls_label_column]
                    if args.generate_mode:
                        input_idss, labels = accelerator.gather((batch['input_ids'], batch['labels']))
                    predictions, references = accelerator.gather((predictions, references))
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(eval_dataloader) - 1:
                            predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                            references = references[: len(eval_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += references.shape[0]
                    predictions = [[int(x) for j, x in enumerate(prediction) if references[i][j] != -100] for i, prediction in enumerate(predictions)]
                    references = [[int(x) for x in reference if x != -100] for reference in references]
                    acc_metric.add_batch(
                        predictions=[i for item in predictions for i in item],
                        references=[i for item in references for i in item],
                    )
                    if args.generate_mode:
                        def remove_negative_100(t):
                            return t[t != -100]
                        for example_index in range(len(predictions)):
                            input_ids = input_idss[example_index]
                            label = labels[example_index]
                            prediction = predictions[example_index]
                            # cls_label = references[example_index]
                            input_ids = [x for x in input_ids if x != tokenizer.pad_token_id]
                            # prediction = [x for i, x in enumerate(prediction) if cls_label[i] != -100]
                            label = [x for x in label if x != -100]
                            sents = tokenizer.decode(input_ids[1:-1]).split(tokenizer.mask_token)
                            highlight = tokenizer.decode(label, skip_special_tokens=True)
                            article_with_importance = [sents[0]]
                            importance_flag = False
                            for i, sent in enumerate(sents[1:]):
                                if prediction[i] == 1:
                                    sent = f'{tokenizer.mask_token} {sent}'
                                article_with_importance.append(sent)
                            article = ' '.join(article_with_importance)
                            test_generate_list.append({'article': article, 'highlights': highlight})
                if args.encoder_prompt == "mask_rouge" or args.encoder_prompt == "mask_rouge_lightweight":
                    labels = batch['labels']
                    mask_gen_kwargs = {
                        "max_length": max_mask_target_length,
                        "num_beams": 1,
                    }
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **mask_gen_kwargs,
                    )
                    generated_tokens = generated_tokens.cpu().numpy()
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    inputs = [prefix + inp for inp in decoded_preds]
                    model_inputs = [
                        tokenizer(input_, max_length=args.max_source_length, padding=padding, truncation=True) for
                        input_ in inputs]
                    batch = second_data_collator(model_inputs)
                    batch["labels"] = labels
                    batch = batch.to(accelerator.device)

                # todo 使用 gold label 测一下二阶段模型的 ROUGE
                if args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
                    labels = batch['labels']
                    mask_labels = batch[mask_labels_ids_name]
                    mask_gen_kwargs = {
                        "max_length": max_mask_target_length,
                        "num_beams": 1,
                    }
                    sep_generated_tokens = accelerator.unwrap_model(sep_model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **mask_gen_kwargs,
                    )

                    sep_generated_tokens_for_test = sep_generated_tokens
                    sep_generated_tokens = sep_generated_tokens.cpu().numpy()
                    if isinstance(sep_generated_tokens, tuple):
                        sep_generated_tokens = sep_generated_tokens[0]
                    sep_decoded_preds = tokenizer.batch_decode(sep_generated_tokens, skip_special_tokens=True)
                    # todo 加上 mask rouge 的测评
                    inputs = [prefix + inp for inp in sep_decoded_preds]
                    model_inputs = [
                        tokenizer(input_, max_length=args.max_source_length, padding=padding, truncation=True) for
                        input_ in inputs]
                    batch = second_data_collator(model_inputs)
                    batch["labels"] = labels
                    batch = batch.to(accelerator.device)

                    sep_generated_tokens = accelerator.pad_across_processes(
                        sep_generated_tokens_for_test, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    if not args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        mask_labels = accelerator.pad_across_processes(mask_labels, dim=1,
                                                                  pad_index=tokenizer.pad_token_id)
                    sep_generated_tokens, mask_labels = accelerator.gather_for_metrics((sep_generated_tokens, mask_labels))
                    sep_generated_tokens = sep_generated_tokens.cpu().numpy()
                    mask_labels = mask_labels.cpu().numpy()
                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        mask_labels = np.where(mask_labels != -100, mask_labels, tokenizer.pad_token_id)
                    if isinstance(sep_generated_tokens, tuple):
                        sep_generated_tokens = sep_generated_tokens[0]

                    sep_decoded_preds = tokenizer.batch_decode(sep_generated_tokens, skip_special_tokens=True)
                    sep_decoded_labels = tokenizer.batch_decode(mask_labels, skip_special_tokens=True)

                    sep_decoded_preds, sep_decoded_labels = postprocess_text(sep_decoded_preds, sep_decoded_labels)

                    sep_metric.add_batch(
                        predictions=sep_decoded_preds,
                        references=sep_decoded_labels,
                    )

                if args.encoder_prompt == "multi_cls_step1":
                    pass
                else:
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
        if args.encoder_prompt == "multi_cls_step1":
            result = acc_metric.compute()
            if args.generate_mode:
                train_generated_dataset = Dataset.from_list(train_generate_list)
                test_generated_dataset = Dataset.from_list(test_generate_list)
                generated_dataset_dict = DatasetDict({"train": train_generated_dataset, "test": test_generated_dataset})
                accelerator.print(generated_dataset_dict["train"][0])
                accelerator.print(generated_dataset_dict)
                generated_dataset_dict = generated_dataset_dict.map(
                    preprocess_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc="Running tokenize on generated dataset",
                )
                generated_dataset_dict.save_to_disk(args.generated_dataset_path)
        else:
            result = metric.compute(use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
        if args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
            sep_result = sep_metric.compute(use_stemmer=True)
            sep_result = {k: round(v * 100, 4) for k, v in sep_result.items()}
            for k, v in sep_result.items():
                result[f'mask_{k}'] = v

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
                if args.encoder_prompt != "multi_cls_step1":
                    result["train_loss"] = total_loss.item() / len(train_dataloader)  # 生成式loss
                if args.encoder_prompt == "multi_cls_step1":
                    result["total_cls_loss"] = total_cls_loss.item() / len(train_dataloader)
                if args.encoder_prompt == "contrastive_kmeans":
                    result["total_contrastive_loss"] = total_contrastive_loss.item() / len(train_dataloader)
                if args.encoder_prompt == "mask_rouge" or args.encoder_prompt == "mask_rouge_lightweight":
                    result["total_mask_loss"] = total_mask_loss.item() / len(train_dataloader)
                result["epoch"] = epoch
                result["step"] = completed_steps
                if args.model_type == "BartForConditionalGenerationWithRouge1":
                    result["total_ext_rouge1_loss"] = total_ext_rouge1_loss.item() / len(train_dataloader)  # rouge 1 loss
                    result["loss_sum"] = result["train_loss"] + result["total_ext_rouge1_loss"]  # 生成式 + rouge 1 loss
                if args.model_type == "BartForConditionalGenerationWithRouge1Rouge2":
                    result["total_ext_rouge1_loss"] = total_ext_rouge1_loss.item() / len(train_dataloader)  # rouge 1 loss
                    result["total_ext_rouge2_loss"] = total_ext_rouge2_loss.item() / len(train_dataloader)  # rouge 2 loss
                    result["loss_sum"] = result["train_loss"] + result["total_ext_rouge1_loss"] + result["total_ext_rouge2_loss"]  # 生成式 + rouge 1 loss + rouge 2 loss
            else:
                accelerator.log(result, step=1)
                accelerator.log(result, step=40000)
            accelerator.log(result, step=completed_steps)

        if args.push_to_hub and epoch < args.num_train_epochs - 1 and not args.test:
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

        if args.checkpointing_steps == "epoch" and not args.test and not args.generate_mode:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None and not args.test and not args.generate_mode:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if args.encoder_prompt == "mask_rouge_lightweight_separate_decoder":
            unwrapped_sep_model = accelerator.unwrap_model(sep_model)
            unwrapped_sep_model.save_pretrained(
                os.path.join(args.output_dir, 'sep'), is_main_process=accelerator.is_main_process, save_function=accelerator.save
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