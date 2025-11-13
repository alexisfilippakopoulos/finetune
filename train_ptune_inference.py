#!/usr/bin/env python3
# train_ptune.py
"""
LoRA fine-tuning training script (patched).
Fixes:
 - token-level masking (no more masking by character length)
 - preserves target tokens where possible, skips examples with no target after truncation
 - explicit BitsAndBytes 4-bit toggle via --load_in_4bit
 - reproducible seed
 - debug logging for mask coverage and skipped examples
"""
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Constants
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_MAP = {
    "summary": "### Input:\n{input}\n### Summary:",
    "question": "### Input:\n{input}\n### Answer:\n{answer}\n### Question:",
    "sentiment": "### Input:\n{input}\n### Sentiment:",
    "inference": "### Input_1:\n{input_1}\n### Input_2:\n{input_2}\n### Inference:",
    "detection": "### Input_1:\n{input_1}\n### Input_2:\n{input_2}\n### Paraphrase Classification:"
}

# -------------------------
# Argument dataclasses
# -------------------------
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class LoRAArguments:
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    target_modules: Optional[str] = field(default=None)
    bias: str = field(default="none")
    load_in_4bit: bool = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)


# -------------------------
# Dataset
# -------------------------
class SupervisedDataset(Dataset):
    """
    Tokenizes source and target separately, concatenates them, and constructs labels
    where source positions are IGNORE_INDEX and target positions contain actual ids.
    Examples that end up with zero target tokens after truncation are skipped.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        shot: int = 2,
        max_len: Optional[int] = None,
    ):
        import utils  # local helper to load json
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        # pick prompt type based on data_path
        prompt_input = None
        prompt_path = None
        for key in PROMPT_MAP:
            if key in data_path:
                prompt_input = PROMPT_MAP[key]
                prompt_path = f"data/{key}/few_shot/multinli_few_shot.json"
                break
        if prompt_input is None:
            # default to summary style if no key found
            prompt_input = PROMPT_MAP["summary"]
            prompt_path = f"data/summary/few_shot/multinli_few_shot.json"

        # load the few-shot template (if exists)
        template = []
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = json.load(f)
        except Exception:
            logging.warning("Could not load prompt_path %s — proceeding without few-shot context", prompt_path)
            template = []

        context = ""
        for tmp in template[:shot]:
            # append context example and its output (keeps same formatting as before)
            context += prompt_input.format_map(tmp) + tmp.get("output", "") + "\n\n"

        sources = [context + prompt_input.format_map(example) for example in list_data_dict]
        targets = [f"{example.get('output','')}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs...")
        max_len = max_len if max_len is not None else tokenizer.model_max_length
        data_dict = self.preprocess(sources, targets, tokenizer, max_len=max_len)

        # store lists of tensors
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.skipped = data_dict.get("skipped", 0)

        logging.warning("Preprocess finished: kept %d examples, skipped %d (no target after truncation)",
                        len(self.input_ids), self.skipped)

    def preprocess(self, sources, targets, tokenizer, max_len: int):
        """
        Tokenize source and target separately and build inputs and labels.
        Strategy:
         - tokenise src and tgt separately
         - if src length >= max_len, truncate src to a tail portion and reserve space for target
         - truncate target to fit remaining space
         - if target becomes empty after truncation, skip example
        """
        input_ids_list = []
        labels_list = []
        skipped = 0

        for idx, (src, tgt) in enumerate(zip(sources, targets)):
            # tokenize source and target separately
            src_toks = tokenizer(src, truncation=True, max_length=max_len, padding=False, return_tensors="pt")
            tgt_toks = tokenizer(tgt, truncation=True, max_length=max_len, padding=False, return_tensors="pt")

            src_ids = src_toks["input_ids"][0]
            tgt_ids = tgt_toks["input_ids"][0]

            # available space for target after keeping source
            avail = max_len - src_ids.shape[0]
            if avail <= 0:
                # source alone too long: keep tail of source to allow room for some target
                # heuristic: keep half of max_len for source, rest for target
                keep_src = max_len // 2
                src_ids = src_ids[-keep_src:]
                avail = max_len - src_ids.shape[0]

            # truncate target to available space
            tgt_ids = tgt_ids[:avail]

            if tgt_ids.shape[0] == 0:
                # nothing left of target after truncation — skip this example
                skipped += 1
                continue

            combined = torch.cat([src_ids, tgt_ids], dim=0)
            # labels: -100 for source positions, actual ids for target positions
            labels = torch.full_like(combined, fill_value=IGNORE_INDEX)
            labels[src_ids.shape[0]:] = combined[src_ids.shape[0]:]

            input_ids_list.append(combined)
            labels_list.append(labels)

        # Debug/log statistics (first few items)
        if len(labels_list) > 0:
            masked_counts = [int((lbl == IGNORE_INDEX).sum().item()) for lbl in labels_list[:5]]
            lengths = [lbl.shape[0] for lbl in labels_list[:5]]
            for i, (l, m) in enumerate(zip(lengths, masked_counts)):
                logging.debug("example %d preview: len=%d masked=%d frac_masked=%.3f", i, l, m, m / l)
        else:
            logging.debug("No examples retained after preprocess (all skipped).")

        return dict(input_ids=input_ids_list, labels=labels_list, skipped=skipped)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# -------------------------
# Data collator
# -------------------------
@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [i["input_ids"] for i in instances]
        labels = [i["labels"] for i in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


# -------------------------
# LoRA helper
# -------------------------
def _parse_target_modules(s: Optional[str]) -> list[str]:
    if s is None:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "fc1", "fc2", "proj"]
    return [x.strip() for x in s.split(",") if x.strip()]


# -------------------------
# Training
# -------------------------
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # reproducible
    transformers.set_seed(42)

    # Load tokenizer first
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Add special tokens if missing
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)

    # BitsAndBytes / 4-bit config: only create if requested
    bnb_config = None
    if lora_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    print("Loading base model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        cache_dir=training_args.cache_dir,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings to match tokenizer if we added tokens
    model.resize_token_embeddings(len(tokenizer))
    logging.warning(f"Tokenizer vocab size: {len(tokenizer)}")

    # Gradient checkpointing to save GPU memory
    model.gradient_checkpointing_enable()

    # warn about potential instability when combining GC + 4-bit
    if lora_args.load_in_4bit:
        logging.warning("Using 4-bit quantization (BitsAndBytes). Ensure this is intended. "
                        "Gradient checkpointing + 4-bit can be unstable in some setups.")

    # Prepare model for k-bit training and LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=_parse_target_modules(lora_args.target_modules),
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Verify trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.warning("Trainable params: %d/%d (%.6f%%)", trainable_params, total_params,
                    100.0 * trainable_params / max(total_params, 1))

    # Dataset & collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, shot=2,
                                      max_len=training_args.model_max_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    # If many examples were skipped during preprocessing, warn
    if getattr(train_dataset, "skipped", 0) > 0:
        logging.warning("Preprocess skipped %d examples (target truncated to zero). Consider increasing model_max_length or reducing few-shot 'shot'.",
                        train_dataset.skipped)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    # Train & save LoRA adapters
    trainer.train()
    peft_save_dir = training_args.output_dir + "_lora"
    model.save_pretrained(peft_save_dir)
    logging.warning("Saved LoRA adapters to: %s", peft_save_dir)


if __name__ == "__main__":
    train()

