
#!/usr/bin/env python3
"""
Create judgmental dataset in Alpaca format for Axolotl training
"""

import json
import os
import sys
from typing import List, Dict

def load_generated_examples() -> List[Dict]:
    """Load all generated synthetic examples from intermediate files."""
    examples = []
    intermediate_dir = "intermediate"
    
    # Load all synthetic_by_seed files
    for filename in os.listdir(intermediate_dir):
        if filename.startswith("synthetic_by_seed_") and filename.endswith(".json"):
            filepath = os.path.join(intermediate_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    seed_examples = json.load(f)
                    examples.extend(seed_examples)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    return examples

def create_alpaca_format(examples: List[Dict]) -> List[Dict]:
    """Convert examples to Alpaca format."""
    alpaca_data = []
    
    for example in examples:
        # Each example is already in Alpaca format with instruction, input, output
        alpaca_entry = {
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "output": example.get("output", ""),
            "meta": example.get("meta", {})
        }
        alpaca_data.append(alpaca_entry)
    
    return alpaca_data

def save_dataset(data: List[Dict], output_file: str):
    """Save dataset in JSONL format for Axolotl."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # Remove meta for training (keep only instruction, input, output)
            training_item = {
                "instruction": item["instruction"],
                "input": item["input"], 
                "output": item["output"]
            }
            f.write(json.dumps(training_item, ensure_ascii=False) + '\n')

def main():
    """Main function to create judgmental Alpaca dataset."""
    print("Loading generated synthetic examples...")
    examples = load_generated_examples()
    
    if not examples:
        print("No synthetic examples found. Please run synthetic generation first.")
        print("Run: python generate_synthetic.py --target-size 200")
        sys.exit(1)
    
    print(f"Found {len(examples)} examples")
    
    # Filter and balance the dataset
    arabic_examples = [ex for ex in examples if ex.get('meta', {}).get('language') == 'ar']
    english_examples = [ex for ex in examples if ex.get('meta', {}).get('language') == 'en']
    
    correct_examples = [ex for ex in examples if ex.get('meta', {}).get('type') == 'correct']
    incorrect_examples = [ex for ex in examples if ex.get('meta', {}).get('type') == 'incorrect']
    
    print(f"Language distribution: Arabic={len(arabic_examples)}, English={len(english_examples)}")
    print(f"Type distribution: Correct={len(correct_examples)}, Incorrect={len(incorrect_examples)}")
    
    # Create Alpaca format
    alpaca_data = create_alpaca_format(examples)
    
    # Save training dataset
    output_file = "data/judgmental_training_dataset.jsonl"
    save_dataset(alpaca_data, output_file)
    print(f"Saved {len(alpaca_data)} examples to {output_file}")
    
    # Create train/val/test splits
    import random
    random.seed(42)
    shuffled = alpaca_data.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    
    # Save splits
    save_dataset(train_data, "data/judgmental_train.jsonl")
    save_dataset(val_data, "data/judgmental_val.jsonl")
    save_dataset(test_data, "data/judgmental_test.jsonl")
    
    print(f"Created splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create Axolotl config
    axolotl_config = {
        "base_model": "microsoft/DialoGPT-medium",
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        
        "load_in_8bit": False,
        "load_in_4bit": True,
        "strict": False,
        
        "datasets": [
            {
                "path": "data/judgmental_train.jsonl",
                "type": "alpaca"
            }
        ],
        
        "val_set_size": 0.0,
        "eval_dataset": [
            {
                "path": "data/judgmental_val.jsonl", 
                "type": "alpaca"
            }
        ],
        
        "dataset_prepared_path": "prepared_data/judgmental",
        "output_dir": "outputs/judgmental_model",
        
        "sequence_len": 2048,
        "sample_packing": False,
        "pad_to_sequence_len": True,
        
        "adapter": "lora",
        "lora_model_dir": "",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj",
            "v_proj", 
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
        
        "wandb_project": "judgmental_bilingual",
        "wandb_entity": "",
        "wandb_watch": "",
        "wandb_name": "",
        "wandb_log_model": "",
        
        "gradient_accumulation_steps": 8,
        "micro_batch_size": 2,
        "num_epochs": 3,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 1e-5,
        
        "train_on_inputs": False,
        "group_by_length": True,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        
        "gradient_checkpointing": True,
        "early_stopping_patience": "",
        "resume_from_checkpoint": "",
        "local_rank": "",
        
        "logging_steps": 10,
        "xformers_attention": "",
        "flash_attention": True,
        
        "warmup_steps": 100,
        "evals_per_epoch": 2,
        "save_steps": "",
        "eval_steps": 100,
        "save_total_limit": 3,
        
        "special_tokens": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>", 
            "pad_token": "[PAD]"
        }
    }
    
    with open("axolotl_judgmental_config.yaml", 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(axolotl_config, f, default_flow_style=False, allow_unicode=True)
    
    print("Created axolotl_judgmental_config.yaml for training")
    print("\nTo train the model with Axolotl:")
    print("1. Install Axolotl: pip install axolotl")  
    print("2. Run training: axolotl train axolotl_judgmental_config.yaml")

if __name__ == "__main__":
    main()
