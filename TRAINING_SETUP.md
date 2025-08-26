# Training Setup Guide: 4B Model on T4 GPU

## Overview
This guide provides complete instructions for fine-tuning a 4B parameter model on Islamic finance bilingual Q&A data using Axolotl and a T4 GPU.

## Hardware Requirements
- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 50GB+ free space for model, dataset, and checkpoints

## Dataset Information
Your converted Alpaca format datasets:
- **Arabic Q&A**: 605 entries (305.2 KB) - `output/arabic_qa_alpaca.jsonl`
- **English Q&A**: 555 entries (221.9 KB) - `output/english_qa_alpaca.jsonl`
- **Bilingual Combined**: 1,078 entries (490.1 KB) - `output/aligned_qa_alpaca.jsonl`

## Model Selection: Microsoft Phi-3.5-mini (3.8B parameters)
**Why this model:**
- Optimal size for T4 GPU (3.8B parameters ≈ 4B target)
- Strong instruction-following capabilities
- Excellent multilingual support (Arabic + English)
- Efficient architecture for memory-constrained training

## Installation Steps

### 1. Install Axolotl
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Axolotl
pip install "axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl.git"

# Install additional dependencies
pip install bitsandbytes accelerate
```

### 2. Prepare Training Data
```bash
# Your Alpaca format files are ready in output/
# Primary training file: output/aligned_qa_alpaca.jsonl
```

## Training Configuration Details

### Memory Optimization for T4
```yaml
load_in_4bit: true              # Essential for 16GB VRAM
micro_batch_size: 1             # Smallest possible batch
gradient_accumulation_steps: 32  # Effective batch size = 32
gradient_checkpointing: true     # Trade compute for memory
bf16: auto                      # Mixed precision training
```

### LoRA Settings for Efficient Fine-tuning
```yaml
adapter: lora
lora_r: 32        # Good balance of capacity vs efficiency
lora_alpha: 16    # Conservative scaling
lora_dropout: 0.05 # Prevent overfitting
```

### Training Hyperparameters
```yaml
learning_rate: 0.0002           # Conservative rate for stability
num_epochs: 3                   # Sufficient for 1K samples
sequence_len: 2048              # Handle long Islamic finance content
optimizer: adamw_8bit           # Memory-efficient optimizer
```

## Training Commands

### Start Training
```bash
# Navigate to your project directory
cd /path/to/your/project

# Start training with the config file
accelerate launch -m axolotl.cli.train axolotl_config.yaml
```

### Monitor Training
```bash
# Monitor GPU usage
nvidia-smi -l 1

# View training logs
tail -f ./fine-tuned-islamic-finance-4b/trainer_state.json
```

### Resume from Checkpoint
```bash
# If training was interrupted
accelerate launch -m axolotl.cli.train axolotl_config.yaml --resume_from_checkpoint ./fine-tuned-islamic-finance-4b/checkpoint-XXX
```

## Expected Training Time
- **Estimated Duration**: 3-4 hours on T4 GPU
- **Total Steps**: ~150-200 steps (depending on data packing)
- **Checkpoints**: Saved every 200 steps

## Memory Usage Estimates
- **Model Loading**: ~8GB VRAM (4-bit quantization)
- **Training Peak**: ~14GB VRAM
- **Available Headroom**: ~2GB for system processes

## Post-Training Steps

### 1. Convert to Standard Format
```bash
# Convert LoRA weights to full model
python -m axolotl.cli.merge_lora axolotl_config.yaml --lora_model_dir ./fine-tuned-islamic-finance-4b
```

### 2. Test the Model
```bash
# Interactive testing
python -m axolotl.cli.inference axolotl_config.yaml --lora_model_dir ./fine-tuned-islamic-finance-4b
```

### 3. Export for Production
```bash
# Save final model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./fine-tuned-islamic-finance-4b')
tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-islamic-finance-4b')
model.save_pretrained('./final-islamic-finance-model')
tokenizer.save_pretrained('./final-islamic-finance-model')
"
```

## Sample Alpaca Training Entries

### Arabic Example
```json
{
  "instruction": "ما هي الجهة المسؤولة عن إصدار كتاب \"المعايير الشرعية\" وما هي أبرز معلومات الاتصال بها؟",
  "input": "",
  "output": "الجهة المسؤولة عن إصدار \"المعايير الشرعية\" هي هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية (أيوفي). معلومات الاتصال تشمل الهاتف: (+973) 17250194، والبريد الإلكتروني: info@aaoifi.com، والموقع الإلكتروني: www.aaoifi.com."
}
```

### English Example
```json
{
  "instruction": "What is the primary governing principle for the Islamic finance industry as stated in the text?",
  "input": "",
  "output": "The primary governing principle for the Islamic finance industry is adherence to and compliance with the rules and principles of Shari'ah, based on collective reasoning by prominent Shari'ah scholars and jurists, and derived from the Qur'an and the Sunnah."
}
```

## Troubleshooting

### Out of Memory Errors
```yaml
# Further reduce batch size
micro_batch_size: 1
gradient_accumulation_steps: 64

# Reduce sequence length
sequence_len: 1024

# Enable more aggressive optimizations
dataloader_pin_memory: false
remove_unused_columns: true
```

### Slow Training
```yaml
# Enable sample packing
sample_packing: true

# Increase dataloader workers (if CPU allows)
dataloader_num_workers: 4
```

### Model Quality Issues
```yaml
# Increase LoRA rank
lora_r: 64

# More epochs
num_epochs: 5

# Lower learning rate
learning_rate: 0.0001
```

## Expected Results
After training, your model should be able to:
- Answer Islamic finance questions in both Arabic and English
- Maintain proper Arabic grammar and formal English style
- Understand complex Sharia compliance topics
- Provide contextually appropriate responses for banking and finance scenarios

## File Structure After Training
```
your-project/
├── axolotl_config.yaml                 # Training configuration
├── output/
│   ├── aligned_qa_alpaca.jsonl         # Training data
│   ├── arabic_qa_alpaca.jsonl          # Arabic-only data
│   └── english_qa_alpaca.jsonl         # English-only data
└── fine-tuned-islamic-finance-4b/      # Training outputs
    ├── adapter_config.json             # LoRA configuration
    ├── adapter_model.bin              # Fine-tuned weights
    ├── trainer_state.json             # Training state
    └── training_args.bin              # Training arguments
```

This setup provides a production-ready training pipeline for your bilingual Islamic finance Q&A model on T4 hardware.