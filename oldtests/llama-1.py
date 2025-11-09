# fine_tune_lora_llama2.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import time
from transformers import TrainerCallback

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        steps_done = state.global_step
        total_steps = state.max_steps

        if steps_done > 0:
            time_per_step = elapsed / steps_done
            remaining_steps = total_steps - steps_done
            eta_seconds = remaining_steps * time_per_step

            # format as H:M:S
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            # safely format loss
            loss_val = logs.get("loss")
            if isinstance(loss_val, (int, float)):
                loss_str = f"{loss_val:.4f}"
            else:
                loss_str = str(loss_val)

            print(
                f"[Step {steps_done}/{total_steps}] "
                f"Loss: {loss_str} | "
                f"Elapsed: {elapsed_str} | ETA: {eta_str}"
            )


# --------------------
# Config
# --------------------
# If you have access to LLaMA-2:
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
# If you don't, uncomment one of these instead:
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

OUTPUT_DIR = "./results-llama2-lora"
SEQ_LEN = 128
BATCH_SIZE = 12
GRAD_ACCUM = 2
EPOCHS = 1

# prefer bfloat16 on RTX 40xx, otherwise fall back to fp16 on CUDA
bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
fp16_ok = torch.cuda.is_available()

# --------------------
# Dataset (plain text LM on Yelp reviews)
# --------------------
dataset = load_dataset("yelp_review_full")
text_column = "text"

# --------------------
# Tokenizer
# --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Make sure we have a pad token for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(batch):
    return tokenizer(
        batch[text_column],
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=[c for c in dataset["train"].column_names if c not in ["input_ids", "attention_mask"]])
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
# Subsample train (optional)
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(100_000))

# Collator for causal LM (no MLM)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --------------------
# 4/8-bit loading with BitsAndBytes
# --------------------
# 8-bit is a good default; switch to 4-bit if you need even lower VRAM (slower)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,          # or: load_in_4bit=True
    # load_in_4bit_kwargs if you choose 4-bit:
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
)

# --------------------
# Base model
# --------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
)
# set pad id to avoid trainer/loader issues
model.config.pad_token_id = tokenizer.pad_token_id

# Prepare for k-bit training (fixes layer norms & gradients for quantized base)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# --------------------
# LoRA config
# --------------------
# For LLaMA-family, these targets work well. If you switch to Falcon, use ["query_key_value"].
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # common LLaMA targets
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # sanity check

# --------------------
# Training args
# --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=400,
    logging_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    fp16=(fp16_ok and not bf16_ok),
    bf16=bf16_ok,
    optim="paged_adamw_8bit",   # needs bitsandbytes; set to "adamw_torch" if you prefer
    save_total_limit=2,
    report_to="none",           # set "tensorboard" if you want TB logs
)

# --------------------
# Trainer
# --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collator,
    callbacks=[ProgressCallback],   # ✅ put callback here
)


trainer.train()
#trainer.train(resume_from_checkpoint="./results-llama2-lora/checkpoint-10")

# Optional: save adapters only (tiny file)
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "lora-adapter"))
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Finished. LoRA adapter saved to:", os.path.join(OUTPUT_DIR, "lora-adapter"))
