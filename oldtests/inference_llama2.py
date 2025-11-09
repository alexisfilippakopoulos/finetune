# inference_lora_llama2.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --------------------
# Config
# --------------------
BASE_MODEL = "meta-llama/Llama-2-7b-hf"   # same as you trained on
ADAPTER_PATH = "./results-llama2-lora/lora-adapter"  # where trainer.model.save_pretrained saved LoRA weights

# --------------------
# Load tokenizer
# --------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------
# Load base model with quantization (optional)
# --------------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# --------------------
# Load LoRA adapter
# --------------------
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# --------------------
# Inference function
# --------------------
def chat(prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --------------------
# Example usage
# --------------------
if __name__ == "__main__":
    user_prompt = "Write a short descripiton for athens."
    response = chat(user_prompt, max_new_tokens=100)
    print("\n=== Model response ===\n")
    print(response)
