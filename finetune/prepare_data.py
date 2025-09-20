import json
import os
import argparse
from typing import List, Dict, Any


def convert_docvqa_to_llava_format(input_file: str, output_file: str, image_folder: str):
    """Convert DocVQA dataset format to LLaVA conversation format"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    
    for item in data:
        # Extract question and answers
        question = item.get('question', '')
        answers = item.get('answers', [])
        
        # Use the first answer as ground truth (or combine if multiple)
        answer = answers[0] if answers else ""
        if len(answers) > 1:
            # Optionally combine multiple answers
            answer = ", ".join(answers)
        
        # Get image filename
        image_filename = item.get('image', '')
        if not image_filename:
            continue
            
        # Create conversation format
        conversation = {
            "id": item.get('questionId', str(len(formatted_data))),
            "image": image_filename,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{question} <image>"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        formatted_data.append(conversation)
    
    # Save formatted data
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(formatted_data)} samples from {input_file} to {output_file}")
    print(f"Image folder: {image_folder}")


def convert_chartvqa_to_llava_format(input_file: str, output_file: str, image_folder: str):
    """Convert ChartVQA dataset format to LLaVA conversation format"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    
    for item in data:
        # Extract question and answer
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # Get image filename
        image_filename = item.get('image', '')
        if not image_filename:
            continue
        
        # Create conversation format
        conversation = {
            "id": item.get('id', str(len(formatted_data))),
            "image": image_filename,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{question} <image>"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        formatted_data.append(conversation)
    
    # Save formatted data
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(formatted_data)} samples from {input_file} to {output_file}")
    print(f"Image folder: {image_folder}")


def convert_general_vqa_format(input_file: str, output_file: str, image_folder: str):
    """Convert general VQA format to LLaVA conversation format"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    
    for i, item in enumerate(data):
        # Try to extract common field names
        question = item.get('question') or item.get('Question') or item.get('query', '')
        answer = item.get('answer') or item.get('Answer') or item.get('response', '')
        
        # Get image filename - try common field names
        image_filename = (
            item.get('image') or 
            item.get('Image') or 
            item.get('image_name') or 
            item.get('filename', '')
        )
        
        if not image_filename:
            print(f"Warning: No image found for sample {i}")
            continue
        
        # Create conversation format
        conversation = {
            "id": item.get('id', str(i)),
            "image": image_filename,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{question} <image>"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        formatted_data.append(conversation)
    
    # Save formatted data
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(formatted_data)} samples from {input_file} to {output_file}")
    print(f"Image folder: {image_folder}")


def main():
    parser = argparse.ArgumentParser(description="Prepare VQA datasets for LLaVA fine-tuning")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON file with VQA data")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON file in LLaVA format")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to directory containing images")
    parser.add_argument("--dataset_type", type=str, 
                        choices=["docvqa", "chartvqa", "general"], default="general",
                        help="Type of dataset")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    # Check if image folder exists
    if not os.path.exists(args.image_folder):
        print(f"Warning: Image folder {args.image_folder} does not exist")
        print("Please make sure the image paths in the dataset are correct")
    
    # Convert based on dataset type
    if args.dataset_type == "docvqa":
        convert_docvqa_to_llava_format(args.input_file, args.output_file, args.image_folder)
    elif args.dataset_type == "chartvqa":
        convert_chartvqa_to_llava_format(args.input_file, args.output_file, args.image_folder)
    else:
        convert_general_vqa_format(args.input_file, args.output_file, args.image_folder)
    
    print("\nData preparation completed!")
    print("Next steps:")
    print(f"1. Verify the output file: {args.output_file}")
    print(f"2. Make sure images are in: {args.image_folder}")
    print("3. Run fine-tuning with: python llava_finetune.py --data_path <output_file> --image_folder <image_folder>")


if __name__ == "__main__":
    main()