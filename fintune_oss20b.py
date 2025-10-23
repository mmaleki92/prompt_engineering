# -*- coding: utf-8 -*-
# --- Full Fine-tuning Script for gpt-oss-20b with Unsloth ---

# --- 1. Installation ---
# Ensure necessary libraries are installed (run this cell first in Colab)
# %%capture
import os, importlib.util
# !pip install --upgrade -qqq uv # Already done if running previous cells
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy, PIL; get_numpy = f"numpy=={numpy.__version__}"; get_pil = f"pillow=={PIL.__version__}"
    except: get_numpy = "numpy"; get_pil = "pillow"
    # Note: transformers==4.56.2 is specified in the original code, ensure compatibility
    # If issues arise, check Unsloth docs for recommended versions.
    print("Installing necessary packages...")
    os.system(f"""
    uv pip install -qqq \\
        "torch>=2.8.0" "triton>=3.4.0" {get_numpy} {get_pil} torchvision bitsandbytes "transformers==4.56.2" \\
        datasets "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \\
        git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels \\
        "trl>=0.22.0" # Ensure TRL version supports needed features
    """)
    print("Installation complete.")
elif importlib.util.find_spec("unsloth") is None:
    print("Installing Unsloth...")
    os.system("uv pip install -qqq unsloth[base]")
    print("Installation complete.")


# --- 2. Imports ---
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TextStreamer
import gc # For garbage collection



# --- 3. Configuration ---
model_name = "unsloth/gpt-oss-20b" # Or "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
max_seq_length = 2048  # Choose based on your VRAM and data, e.g., 1024, 2048
load_in_4bit = True   # Use 4-bit quantization
dtype = None          # Auto detection
lora_r = 8            # LoRA rank
lora_alpha = 16       # LoRA alpha
per_device_train_batch_size = 1 # Keep low for 20B on T4/free Colab
gradient_accumulation_steps = 8 # Effective batch size = batch_size * grad_accum
num_train_epochs = 2  # Number of passes over the dataset (1-3 recommended)
learning_rate = 2e-4
output_dir = "outputs"
data_file = "finetune_data.jsonl" # Your exported JSONL file



# --- 4. Load Model and Tokenizer ---
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    dtype = dtype,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # Use if loading gated models
)
print("Model and tokenizer loaded.")



# --- 5. Add LoRA Adapters ---
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_r,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # Standard targets for many models
    lora_alpha = lora_alpha,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Crucial for memory saving
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
print("LoRA adapters added.")



# --- 6.
print("Loading and preparing dataset...")

# Define the formatting function (Keep this)
def formatting_prompts_func(examples):
    messages = examples["messages"]
    # Apply chat template to each conversation in the batch
    texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages]
    return { "text": texts, }

# Load your dataset from the JSONL file (Keep this)
try:
    dataset = load_dataset("json", data_files=data_file, split="train")
except Exception as e:
    print(f"Error loading dataset '{data_file}': {e}")
    print("Please ensure the file exists and is a valid JSONL.")
    exit()

# Apply the formatting function to create the "text" column (Keep this)
formatted_dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    desc="Formatting prompts",
)
print(f"Dataset structure after formatting: {formatted_dataset}")



# --- 7. Configure Trainer ---
print("Configuring SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,  # <<< Pass the dataset with the 'text' column
    dataset_text_field = "text",       # <<< Tell trainer where the formatted text is
    max_seq_length = max_seq_length,
    # packing = True, # <<< REMOVE or set to False. Packing often requires pre-tokenization.
                      # Let's start without it for simplicity when using assistant_only_loss.
    dataset_num_proc = 2,
    args = SFTConfig(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 10,
        num_train_epochs = num_train_epochs,
        # max_steps = 30, # Use max_steps for quick testing instead of epochs
        learning_rate = learning_rate,
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to = "none",
        save_strategy = "epoch",
        # save_steps = 100,
        assistant_only_loss = True, # <<< Keep this, it should work now
        # bf16 = torch.cuda.is_bf16_supported(),
        # tf32 = torch.cuda.is_tf32_supported(),
    ),
)
print("Trainer configured.")



# --- 8. Memory & GPU Check ---
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved before training.")



# --- 9. Train ---
print("Starting training...")
trainer_stats = trainer.train()
print("Training finished.")



# --- 10. Post-Training Stats ---
try:
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
except KeyError:
    print("Could not retrieve training runtime metrics.")
except Exception as e:
    print(f"An error occurred while printing memory stats: {e}")



# --- 11. Save the Fine-tuned Model (LoRA Adapters) ---
print("Saving LoRA adapters...")
save_directory = "finetuned_model_adapters"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory) # Save tokenizer too
print(f"LoRA adapters saved to {save_directory}")

# --- Optional: Clean up memory ---
del model
del trainer
del dataset
del formatted_dataset
del tokenized_dataset
gc.collect()
torch.cuda.empty_cache()
gc.collect()
print("Cleaned up memory.")


# --- 12. (Optional) Load and Run Inference ---
run_inference = False # Set to True to run a quick test after saving

if run_inference:
    print("\n--- Running Inference Test ---")
    try:
        from unsloth import FastLanguageModel # Re-import if needed

        # Load the base model with the saved LoRA adapters merged
        print("Loading fine-tuned model for inference...")
        inf_model, inf_tokenizer = FastLanguageModel.from_pretrained(
            model_name = save_directory, # Load from where you saved adapters
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(inf_model) # Optimize for inference
        print("Model loaded for inference.")

        # Example inference prompt (modify as needed)
        messages = [
            # IMPORTANT: For inference, DON'T include the developer prompt
            # if you baked it in during training.
            {"role": "user", "content": "Hi, I need a camera for wildlife, not too heavy."},
            # Add more history if needed for context testing
        ]

        inputs = inf_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Add assistant prompt marker
            return_tensors = "pt",
            return_dict = True,
            reasoning_effort = "medium", # Choose reasoning effort
        ).to("cuda")

        print("\nGenerating response...")
        streamer = TextStreamer(inf_tokenizer)
        _ = inf_model.generate(
            **inputs,
            max_new_tokens = 128, # Adjust as needed
            streamer = streamer,
            use_cache = True,
            # Common generation parameters (optional)
            # temperature=0.7,
            # top_p=0.9,
            # do_sample=True,
            )
        print("\nInference complete.")

    except Exception as e:
        print(f"Error during inference test: {e}")
        print("Skipping inference.")

print("\n--- Script Finished ---")