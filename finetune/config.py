# Configuration file for LLaVA LoRA fine-tuning

# Base model configurations
MODEL_CONFIGS = {
    "llava-v1.5-7b": {
        "model_name_or_path": "./checkpoints/llava-v1.5-7b",
        "version": "v1",
        "vision_tower": "openai/clip-vit-large-patch14",
        "model_max_length": 2048
    },
    "llava-v1.5-13b": {
        "model_name_or_path": "./checkpoints/llava-v1.5-13b",
        "version": "v1",
        "vision_tower": "openai/clip-vit-large-patch14",
        "model_max_length": 2048
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    "docvqa": {
        "default_data_path": "/path/to/docvqa/train.json",
        "default_image_folder": "/path/to/docvqa/images",
        "recommended_epochs": 3,
        "recommended_batch_size": 8,
        "recommended_lr": 2e-5
    },
    "chartvqa": {
        "default_data_path": "/path/to/chartvqa/train.json",
        "default_image_folder": "/path/to/chartvqa/images",
        "recommended_epochs": 4,
        "recommended_batch_size": 8,
        "recommended_lr": 1.5e-5
    }
}

# LoRA configurations
LORA_CONFIGS = {
    "default": {
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "small": {
        "lora_r": 32,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_bias": "none"
    },
    "large": {
        "lora_r": 128,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none"
    }
}

# Training configurations
TRAINING_CONFIGS = {
    "default": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "fp16": True,
        "bf16": False,
        "max_grad_norm": 1.0,
        "save_steps": 500,
        "logging_steps": 10,
        "evaluation_strategy": "no",
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 4,
        "lazy_preprocess": True
    },
    "fast": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 3e-5,
        "save_steps": 200,
        "logging_steps": 5
    },
    "precise": {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "save_steps": 1000,
        "logging_steps": 20
    }
}

# Output directory templates
OUTPUT_DIR_TEMPLATES = {
    "llava-docvqa": "./checkpoints/llava-docvqa-lora-{timestamp}",
    "llava-chartvqa": "./checkpoints/llava-chartvqa-lora-{timestamp}",
    "custom": "./checkpoints/{model}-{dataset}-lora-{timestamp}"
}

def get_config(model_name: str, dataset_type: str, config_type: str = "default") -> dict:
    """
    Get a complete configuration for fine-tuning
    
    Args:
        model_name: Name of the base model
        dataset_type: Type of dataset (docvqa, chartvqa)
        config_type: Type of configuration (default, fast, precise)
    
    Returns:
        dict: Complete configuration dictionary
    """
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}")
    
    if config_type not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(TRAINING_CONFIGS.keys())}")
    
    # Merge configurations
    config = {
        **MODEL_CONFIGS[model_name],
        **LORA_CONFIGS["default"],
        **TRAINING_CONFIGS[config_type],
        "dataset_type": dataset_type
    }
    
    # Add dataset-specific recommendations
    dataset_config = DATASET_CONFIGS[dataset_type]
    config.update({
        "default_data_path": dataset_config["default_data_path"],
        "default_image_folder": dataset_config["default_image_folder"],
        "recommended_epochs": dataset_config["recommended_epochs"],
        "recommended_batch_size": dataset_config["recommended_batch_size"],
        "recommended_lr": dataset_config["recommended_lr"]
    })
    
    return config

def print_config_summary(config: dict):
    """Print a summary of the configuration"""
    print("=== Fine-tuning Configuration Summary ===")
    print(f"Model: {config.get('model_name_or_path', 'N/A')}")
    print(f"Dataset: {config.get('dataset_type', 'N/A')}")
    print(f"Epochs: {config.get('num_train_epochs', 'N/A')}")
    print(f"Batch Size: {config.get('per_device_train_batch_size', 'N/A')}")
    print(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"LoRA Rank: {config.get('lora_r', 'N/A')}")
    print(f"LoRA Alpha: {config.get('lora_alpha', 'N/A')}")
    print("========================================")

if __name__ == "__main__":
    # Example usage
    config = get_config("llava-v1.5-7b", "docvqa", "default")
    print_config_summary(config)