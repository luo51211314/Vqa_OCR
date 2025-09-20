#!/usr/bin/env python3

import os
import json
import torch
from transformers import AutoTokenizer, AutoConfig
from llava.model import *


def check_model_compatibility(model_path: str):
    """Check if the model is compatible with LLaVA fine-tuning"""
    
    print(f"Checking model compatibility: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path does not exist: {model_path}")
        return False
    
    try:
        # Check if it's a LLaVA model
        config = AutoConfig.from_pretrained(model_path)
        
        # Check for LLaVA specific attributes
        has_vision_config = hasattr(config, 'vision_config') or hasattr(config, 'mm_vision_tower')
        has_projector = hasattr(config, 'mm_hidden_size') or hasattr(config, 'mm_projector_type')
        
        if has_vision_config and has_projector:
            print("✅ Model appears to be a LLaVA model")
            return True
        else:
            print("⚠️  Model may not be a LLaVA model. Checking if it's a base language model...")
            
            # Check if it's a supported base model
            supported_models = ['llama', 'mistral', 'vicuna', 'opt']
            model_type = getattr(config, 'model_type', '').lower()
            
            if any(supported in model_type for supported in supported_models):
                print("✅ Model is a supported base language model")
                print("ℹ️  Note: You may need to add vision tower separately")
                return True
            else:
                print(f"❌ Model type '{model_type}' may not be supported")
                return False
                
    except Exception as e:
        print(f"❌ Error loading model config: {e}")
        return False


def check_data_compatibility(data_path: str):
    """Check if the data is in compatible format"""
    
    print(f"Checking data compatibility: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file does not exist: {data_path}")
        return False
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Data should be a list of samples")
            return False
        
        valid_samples = 0
        
        for i, sample in enumerate(data):
            # Check required fields
            if 'conversations' in sample:
                # LLaVA format
                conversations = sample['conversations']
                if len(conversations) < 2:
                    print(f"❌ Sample {i}: Need at least 2 turns in conversation")
                    continue
                
                # Check if contains image token
                human_turn = conversations[0]['value']
                if '<image>' not in human_turn and DEFAULT_IMAGE_TOKEN not in human_turn:
                    print(f"⚠️  Sample {i}: No image token found in human turn")
                    print(f"   Human turn: {human_turn[:100]}...")
                
                valid_samples += 1
                
            elif 'question' in sample and 'answer' in sample:
                # Alternative format
                if 'image' not in sample:
                    print(f"❌ Sample {i}: Missing 'image' field")
                    continue
                
                valid_samples += 1
                
            else:
                print(f"❌ Sample {i}: Unknown format. Expected 'conversations' or 'question/answer' fields")
                print(f"   Sample keys: {list(sample.keys())}")
        
        print(f"✅ Found {valid_samples} valid samples out of {len(data)} total")
        
        if valid_samples > 0:
            print("✅ Data format appears compatible")
            return True
        else:
            print("❌ No valid samples found")
            return False
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def check_image_folder(image_folder: str, data_path: str):
    """Check if images referenced in data exist"""
    
    print(f"Checking image folder: {image_folder}")
    
    if not os.path.exists(image_folder):
        print(f"❌ Error: Image folder does not exist: {image_folder}")
        return False
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        image_files = set()
        missing_images = 0
        
        for sample in data:
            if 'image' in sample:
                image_files.add(sample['image'])
            elif 'conversations' in sample:
                # Try to extract image from conversation context
                pass
        
        # Check if images exist
        for image_file in list(image_files)[:10]:  # Check first 10 images
            image_path = os.path.join(image_folder, image_file)
            if not os.path.exists(image_path):
                print(f"⚠️  Image not found: {image_path}")
                missing_images += 1
        
        if missing_images > 0:
            print(f"⚠️  Found {missing_images} missing images (sampled first 10)")
            print("ℹ️  This may indicate path issues, but training can continue")
        else:
            print("✅ All sampled images found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking images: {e}")
        return False


def check_gpu_memory():
    """Check available GPU memory"""
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - will use CPU")
        return True
    
    try:
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3  # GB
            free_memory = torch.cuda.mem_get_info(i)[0] / 1024**3  # GB
            
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {total_memory:.1f} GB")
            print(f"  Free Memory: {free_memory:.1f} GB")
            
            # Rough estimate of memory requirements
            if free_memory < 4:
                print("⚠️  Low GPU memory - consider reducing batch size")
            elif free_memory >= 16:
                print("✅ Sufficient GPU memory for training")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check compatibility for LLaVA fine-tuning")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to image folder")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLaVA Fine-tuning Compatibility Check")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("Model Compatibility", lambda: check_model_compatibility(args.model_path)),
        ("Data Compatibility", lambda: check_data_compatibility(args.data_path)),
        ("Image Folder", lambda: check_image_folder(args.image_folder, args.data_path)),
        ("GPU Memory", check_gpu_memory)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * len(check_name))
        
        try:
            result = check_func()
            if not result:
                all_passed = False
                print("❌ Check failed")
            else:
                print("✅ Check passed")
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All compatibility checks passed!")
        print("You can proceed with fine-tuning.")
    else:
        print("⚠️  Some compatibility checks failed.")
        print("Please address the issues above before proceeding.")
    print("=" * 60)


if __name__ == "__main__":
    main()'