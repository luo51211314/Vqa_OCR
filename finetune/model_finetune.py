import os
import argparse
import torch
import transformers
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import json


def load_vqa_dataset(data_path, image_folder):
    """Load VQA dataset from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to format expected by training
    formatted_data = []
    for item in data:
        if 'conversations' in item:
            # Already in conversation format
            formatted_data.append(item)
        else:
            # Convert to conversation format
            conversation = {
                "id": item.get("id", "unknown"),
                "image": item.get("image", ""),
                "conversations": [
                    {
                        "from": "human",
                        "value": item.get("question", "") + " " + DEFAULT_IMAGE_TOKEN
                    },
                    {
                        "from": "gpt", 
                        "value": item.get("answer", "")
                    }
                ]
            }
            formatted_data.append(conversation)
    
    return formatted_data


def create_lora_config(model, r=64, alpha=16, dropout=0.05):
    """Create LoRA configuration"""
    # Find all linear layers for LoRA
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Skip certain layers
            if any(keyword in name for keyword in ['lm_head', 'vision_tower', 'mm_projector']):
                continue
            target_modules.append(name.split('.')[-1])
    
    # Remove duplicates
    target_modules = list(set(target_modules))
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    return lora_config


def main():
    parser = argparse.ArgumentParser(description="General Model LoRA Fine-tuning for VQA Tasks")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--model_type", type=str, choices=["llama", "mistral", "other"], default="llama",
                        help="Type of base model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to directory containing images")
    parser.add_argument("--dataset_type", type=str, choices=["docvqa", "chartvqa", "general"], required=True,
                        help="Type of dataset")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/model-finetune-lora",
                        help="Directory to save the model")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    
    # Other arguments
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    
    args = parser.parse_args()
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = create_lora_config(
        model, 
        r=args.lora_r, 
        alpha=args.lora_alpha, 
        dropout=args.lora_dropout
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    dataset = load_vqa_dataset(args.data_path, args.image_folder)
    
    # TODO: Implement training loop
    # This would include data preprocessing, training, and saving
    
    print("Training setup complete. Implement training loop based on your specific needs.")
    print(f"Model will be saved to: {args.output_dir}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


if __name__ == "__main__":
    main()