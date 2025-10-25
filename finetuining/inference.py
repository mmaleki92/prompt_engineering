import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import gc

# --- Configuration ---
hub_model_id = "mmaleki92/gpt-oss-20b-finetuned-instagram_v01"
max_seq_length = 2048 # Should match training
load_in_4bit = True
dtype = None

# --- Load Model and Tokenizer FROM HUB ---
print(f"Loading fine-tuned model from Hub: {hub_model_id}")
# Unsloth automatically downloads the base gpt-oss-20b and applies your adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = hub_model_id, # <<< Point to your Hub repo
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "YOUR_HF_READ_TOKEN", # Add token if the repo is PRIVATE
)
print("Model loaded successfully from Hub.")

# Optimize for Inference
FastLanguageModel.for_inference(model)
print("Model optimized for inference.")

# Optional: Clean up memory
gc.collect()
torch.cuda.empty_cache()
gc.collect()

# --- Run Inference (Example) ---
print("\n--- Running Inference Test ---")
messages = [
    # No developer prompt needed if baked in!
    {"role": "user", "content": "سلام، وقت بخیر"} # Example user message
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "medium", # Choose reasoning effort
).to("cuda")

streamer = TextStreamer(tokenizer, skip_prompt=True)

print("\nGenerating response...")
_ = model.generate(
    **inputs,
    max_new_tokens = 150, # Adjust as needed
    streamer = streamer,
    use_cache = True,
    # Add generation params like temperature, top_p if desired
)
print("\n--- Inference Finished ---")