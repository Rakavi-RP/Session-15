import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from model import create_model_from_config
import yaml

# Get the config file path - adjust for Kaggle input directory
config_path = os.path.join("/kaggle/input/configfile", "config_smollm2_135M.yaml")

# Load config
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    padding_side="right",
    truncation_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# Update config with correct vocab size
config_dict["model"]["model_config"]["vocab_size"] = len(tokenizer)

# Set hyperparameters (matching oldtrain.py)
BATCH_SIZE = 8
MAX_SEQ_LEN = 128
NUM_TRAIN_STEPS = 12001
LEARNING_RATE = 0.0003

# Get device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = create_model_from_config(config_dict)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model = model.to(device)

# Dataset setup
class StreamingTextDatasetWrapper(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __iter__(self):
        for sample in self.hf_dataset:
            text = sample["text"]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            if encoding["input_ids"].shape[1] == self.max_length:
                yield encoding["input_ids"].squeeze(0)

# Setup data
train_dataset = StreamingTextDatasetWrapper(hf_dataset, tokenizer, MAX_SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.01
)

best_loss = float('inf')
best_model_state = None
global_step = 0

model.train()
print("Starting training...")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop (simplified like oldtrain.py)
while global_step < NUM_TRAIN_STEPS:
    for batch in train_dataloader:
        try:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass
            logits = model(inputs)
            
            # Update MoE routing biases (keeping load balancing)
            if hasattr(model, 'module'):  # If using DataParallel
                for layer in model.module.layers:
                    if hasattr(layer.moe, 'expert_load') and layer.moe.expert_load is not None:
                        layer.moe.update_bias_terms(layer.moe.expert_load)
            else:
                for layer in model.layers:
                    if hasattr(layer.moe, 'expert_load') and layer.moe.expert_load is not None:
                        layer.moe.update_bias_terms(layer.moe.expert_load)
            
            # Compute loss
            logits = logits.reshape(-1, config_dict["model"]["model_config"]["vocab_size"])
            targets = targets.reshape(-1)
            loss = criterion(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Print progress
            print(f"Step {global_step}/{NUM_TRAIN_STEPS} | Loss: {loss.item():.4f}")
            
            # Generate sample text periodically
            if global_step % 2000 == 0 or global_step == NUM_TRAIN_STEPS - 1:
                print(f"\n=== Generating sample at step {global_step} ===")
                model.eval()
                with torch.no_grad():
                    # Prepare prompt
                    prompt = "Solar system is a"
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    
                    # Generate with better sampling
                    temperature = 1.0
                    top_p = 0.9
                    repetition_penalty = 1.5
                    max_length = 30
                    generated = input_ids
                    
                    for _ in range(max_length):
                        outputs = model(generated)
                        next_token_logits = outputs[:, -1, :]
                        # Apply repetition penalty
                        for token in set(generated[0].tolist()):
                            next_token_logits[0, token] /= repetition_penalty

                        next_token_logits = next_token_logits / temperature
                        
                        # Apply top-p (nucleus) sampling
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                        
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated = torch.cat([generated, next_token], dim=1)
                        
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                    
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    print(f"Prompt: {prompt}")
                    print(f"Generated: {generated_text}")
                    print("=" * 50)
                
                model.train()
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict()
                print(f"New best loss: {best_loss:.4f} at step {global_step}")
                torch.save(best_model_state, 'best_model.pt')
            
            global_step += 1
            if global_step > NUM_TRAIN_STEPS:
                break
                
        except Exception as e:
            print(f"Error in batch: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

print("\nTraining complete!")
print(f"Best loss achieved: {best_loss:.4f}")
