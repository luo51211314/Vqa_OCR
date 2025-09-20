import os
import argparse
import torch
import transformers
from transformers import TrainingArguments
from llava.model import *
from llava.train.train import train
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token


def main():
    parser = argparse.ArgumentParser(description="LLaVA LoRA Fine-tuning for DocVQA and ChartVQA")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="./checkpoints/llava-v1.5-7b",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--version", type=str, default="v1", help="Model version")
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14",
                        help="Name of vision tower to use")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to directory containing images")
    parser.add_argument("--dataset_type", type=str, choices=["docvqa", "chartvqa"], required=True,
                        help="Type of dataset: docvqa or chartvqa")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/llava-finetune-lora",
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
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        evaluation_strategy="no",
        remove_unused_columns=False,
        report_to=[],
        
        # LoRA specific settings
        lora_enable=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias="none",
        
        # Model settings
        model_max_length=args.max_seq_length,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        lazy_preprocess=True,
    )
    
    # Set up model arguments
    model_args = {
        "model_name_or_path": args.model_name_or_path,
        "version": args.version,
        "vision_tower": args.vision_tower,
        "freeze_backbone": False,
        "tune_mm_mlp_adapter": False,
        "mm_vision_select_layer": -2,
        "mm_use_im_start_end": False,
        "mm_use_im_patch_token": False,
    }
    
    # Set up data arguments
    data_args = {
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "is_multimodal": True,
        "image_aspect_ratio": "pad",
        "lazy_preprocess": True,
    }
    
    print(f"Starting LLaVA LoRA fine-tuning for {args.dataset_type.upper()}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training data: {args.data_path}")
    print(f"Image folder: {args.image_folder}")
    
    # Start training
    train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )


if __name__ == "__main__":
    main()