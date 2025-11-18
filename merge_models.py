from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

base = "meta-llama/Llama-2-7b-hf"
lora_model_path = "/kaggle/input/finetune-sentiment/pytorch/default/1/sentiment/amazon_100/checkpoint-50"
out_path = "/kaggle/working/finetune/saved_models/amazon_100_merged"

# load tokenizer (recommended) — helps keep tokenizer & embeddings in sync
tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)

# load base model (use dtype kwarg not torch_dtype)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")

# helper: try a few likely filenames for the adapter state dict
possible_files = [
    "pytorch_model.bin",
    "adapter_model.bin",
    "adapter_model_state_dict.pt",
    "pytorch_adapter.bin",
    "model.safetensors"  # not loadable by torch.load, but kept for reference
]

adapter_sd = None
adapter_file_used = None
for fn in possible_files:
    path = os.path.join(lora_model_path, fn)
    if os.path.isfile(path):
        try:
            adapter_sd = torch.load(path, map_location="cpu")
            adapter_file_used = path
            break
        except Exception as e:
            # skip files torch can't load (like safetensors)
            continue

# If we did not find a loadable state dict, try to inspect the peft folder structure:
if adapter_sd is None:
    # sometimes PEFT saves the relevant weights under "adapter_model.bin" inside the folder
    # or the folder may already contain a 'pytorch_model.bin' under subfolders; attempt to load direct peft file via torch
    # fallback: try to load the entire folder via PeftModel but we still want to detect vocab mismatch first;
    # If automatic detection fails, we will just try loading PeftModel and catch the error.
    adapter_sd = None

def find_embedding_size_in_state_dict(sd):
    if sd is None:
        return None
    # search keys for something that looks like embed_tokens or lm_head
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            key_lower = k.lower()
            if "embed_tokens.weight" in key_lower or ("lm_head.weight" in key_lower and v.dim() == 2):
                return v.shape[0]
    return None

adapter_vocab_size = find_embedding_size_in_state_dict(adapter_sd)

# If we couldn't detect adapter size from saved files, try a safe attempt: attempt to load PEFT and catch the error to extract expected shape.
if adapter_vocab_size is None:
    try:
        # attempt load — this will raise the size-mismatch RuntimeError you saw; we catch it and parse it
        _ = PeftModel.from_pretrained(model, lora_model_path)
    except RuntimeError as e:
        msg = str(e)
        # parse patterns like "... copying a param with shape torch.Size([32001, 4096]) ... current model is torch.Size([32000, 4096])."
        import re
        m = re.search(r"copying a param with shape torch.Size\(\[(\d+),\s*\d+\]\).*current model is torch.Size\(\[(\d+),\s*\d+\]\)", msg)
        if m:
            adapter_vocab_size = int(m.group(1))
        else:
            # try alternative parse for one of the keys
            m2 = re.search(r"shape torch.Size\(\[(\d+),", msg)
            if m2:
                adapter_vocab_size = int(m2.group(1))

# If we have found a target vocab size from the adapter, compare and resize if needed
if adapter_vocab_size is not None:
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"Adapter vocab size detected: {adapter_vocab_size}; current model vocab size: {current_vocab_size}")
    if adapter_vocab_size != current_vocab_size:
        # If tokenizer is present, it is better to align its length too (optionally add placeholder token)
        # We will resize model embeddings to adapter_vocab_size. If you want tokenizer adjusted instead, add tokens and use their new length.
        print(f"Resizing model embeddings from {current_vocab_size} -> {adapter_vocab_size}")
        model.resize_token_embeddings(adapter_vocab_size)
else:
    print("Could not detect adapter vocab size automatically. Proceeding to load adapter directly (may still fail).")

# Now load PEFT adapter
model = PeftModel.from_pretrained(model, lora_model_path)

# Merge LoRA into base model and unload PEFT wrappers
model = model.merge_and_unload()

# Save merged model
os.makedirs(out_path, exist_ok=True)
model.save_pretrained(out_path)
# Optionally also save tokenizer if you adjusted it (not necessary here unless you added tokens)
tokenizer.save_pretrained(out_path)

print("Merged model saved to", out_path)

