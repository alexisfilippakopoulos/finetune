#Lora Finetuning
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig  # for quantization

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, shot=2):
        import utils
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        # Choose prompt template based on data_path
        for key in PROMPT_MAP:
            if key in data_path:
                prompt_input = PROMPT_MAP[key]
                #Set xsum or socialqa for summary or question
                prompt_path = f"data/{key}/few_shot/xsum_few_shot.json"
                #prompt_path = f"data/{key}/few_shot/socialqa_few_shot.json"

        with open(prompt_path) as f:
            template = json.load(f)

        context = ""
        for tmp in template[:shot]:
            context += prompt_input.format_map(tmp) + tmp["output"] + "\n\n"

        sources = [context + prompt_input.format_map(example) for example in list_data_dict]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs...")
        data_dict = self.preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def preprocess(self, sources, targets, tokenizer):
        examples = [s + t for s, t in zip(sources, targets)]
        tokenized = [tokenizer(e, truncation=True, max_length=tokenizer.model_max_length,
                               padding="longest", return_tensors="pt") for e in examples]
        input_ids = [t["input_ids"][0] for t in tokenized]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, [len(s) for s in sources]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

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
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

# -------------------------
# LoRA helper
# -------------------------
def _parse_target_modules(s: Optional[str]) -> list[str]:
    if s is None:
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj","fc1","fc2","proj"]
    return [x.strip() for x in s.split(",") if x.strip()]

# -------------------------
# Training
# -------------------------
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Load tokenizer first
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                           model_max_length=training_args.model_max_length,
                                                           padding_side="right",
                                                           use_fast=False)
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
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load model in 8-bit if GPU RAM < 16GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if not lora_args.load_in_4bit else None

    print("Loading base model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        cache_dir=training_args.cache_dir,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        
    )
    #Added this line
    model.resize_token_embeddings(len(tokenizer))
    logging.warning(f"Tokenizer vocab size: {len(tokenizer)}")
		
    # Gradient checkpointing to save GPU memory
    model.gradient_checkpointing_enable()

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=_parse_target_modules(lora_args.target_modules),
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # Verify trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.warning(f"Trainable params: {trainable_params}/{total_params} ({100*trainable_params/total_params:.4f}%)")

    # Dataset & collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, shot=2)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)

    # Train & save LoRA adapters
    trainer.train()
    peft_save_dir = training_args.output_dir + "_lora"
    model.save_pretrained(peft_save_dir)
    logging.warning(f"Saved LoRA adapters to: {peft_save_dir}")

if __name__ == "__main__":
    train()
