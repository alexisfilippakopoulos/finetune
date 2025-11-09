import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

import transformers.models.falcon.modeling_falcon as falcon_mod
import torch

# Save original forward
orig_forward = falcon_mod.FalconDecoderLayer.forward

def patched_forward(self, *args, **kwargs):
    # call original forward
    outputs = orig_forward(self, *args, **kwargs)
    # clone any in-place operations that cause autograd errors
    # Falcon internally does mlp_output += attention_output
    # Here we clone attention_output to avoid in-place modification issues
    if hasattr(outputs, "mlp_output"):
        outputs.mlp_output = outputs.mlp_output.clone()
    return outputs

falcon_mod.FalconDecoderLayer.forward = patched_forward

# Load dataset (example: sentiment classification)
dataset = load_dataset("yelp_review_full")

# Load pretrained model & tokenizer
#model_name = "meta-llama/Llama-2-7b-hf"  # Or try "tiiuae/falcon-7b"
model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Fix padding token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Save VRAM
    device_map="auto"
)

# Apply LoRA config (parameter-efficient fine-tuning)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],  # Falcon uses this instead of q_proj/v_proj
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset["train"].train_test_split(test_size=0.1)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",   # or "epoch"
    eval_steps=500,                # how often to eval (if "steps")
    save_strategy="epoch",         # saving checkpoints
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
