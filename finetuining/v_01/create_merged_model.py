import torch
from unsloth import FastLanguageModel
import gc
import os

# --- Configuration ---
local_adapter_directory = "finetuned_model_adapters" # Adapters you just saved
merged_model_directory = "gpt-oss-20b-finetuned-merged-fp16" # Output dir for merged model
max_seq_length = 2048 # Match training
load_in_4bit = True   # Load base in 4bit to apply adapters
dtype = None

print(f"Loading model with adapters from '{local_adapter_directory}'...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_adapter_directory,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    dtype = dtype,
)
print("Model with adapters loaded.")

print(f"Merging adapters and saving to '{merged_model_directory}' in float16...")
os.makedirs(merged_model_directory, exist_ok=True)
# Save in standard float16/bfloat16 for compatibility
model.save_pretrained_merged(merged_model_directory, tokenizer, save_method="merged_16bit")
print(f"Model successfully merged and saved to '{merged_model_directory}'.")

del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
print("Cleaned up memory.")
