#!/bin/bash

# LLaVA LoRA Fine-tuning Script for DocVQA and ChartVQA
# Usage: ./run_finetune.sh [docvqa|chartvqa] [model_path] [data_path] [image_folder] [output_dir]

set -e

# Default parameters
DATASET_TYPE=${1:-docvqa}
MODEL_PATH=${2:-./checkpoints/llava-v1.5-7b}
DATA_PATH=${3:-/path/to/your/dataset/train.json}
IMAGE_FOLDER=${4:-/path/to/your/images}
OUTPUT_DIR=${5:-./checkpoints/llava-${DATASET_TYPE}-lora}

# Training parameters
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5
LORA_R=64
LORA_ALPHA=16
MAX_SEQ_LENGTH=2048

# Validate dataset type
if [[ "$DATASET_TYPE" != "docvqa" && "$DATASET_TYPE" != "chartvqa" ]]; then
    echo "Error: Dataset type must be 'docvqa' or 'chartvqa'"
    exit 1
fi

# Check if files exist
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path '$MODEL_PATH' does not exist"
    exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: Data file '$DATA_PATH' does not exist"
    exit 1
fi

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "Error: Image folder '$IMAGE_FOLDER' does not exist"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==========================================="
echo "LLaVA LoRA Fine-tuning Configuration"
echo "==========================================="
echo "Dataset Type: $DATASET_TYPE"
echo "Model Path: $MODEL_PATH"
echo "Data Path: $DATA_PATH"
echo "Image Folder: $IMAGE_FOLDER"
echo "Output Directory: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "LoRA Rank: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "==========================================="

# Run fine-tuning
python llava_finetune.py \
    --model_name_or_path "$MODEL_PATH" \
    --version "v1" \
    --vision_tower "openai/clip-vit-large-patch14" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --dataset_type "$DATASET_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --save_steps 500 \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --gradient_accumulation_steps 1

echo "Fine-tuning completed!"
echo "Model saved to: $OUTPUT_DIR"

# Save training configuration
cat > "$OUTPUT_DIR/training_config.txt" << EOF
Training Configuration:
=======================
Dataset Type: $DATASET_TYPE
Model: $MODEL_PATH
Data: $DATA_PATH
Image Folder: $IMAGE_FOLDER
Output Dir: $OUTPUT_DIR
Epochs: $NUM_EPOCHS
Batch Size: $BATCH_SIZE
Learning Rate: $LEARNING_RATE
LoRA Rank: $LORA_R
LoRA Alpha: $LORA_ALPHA
Max Sequence Length: $MAX_SEQ_LENGTH
Training Date: $(date)
EOF

echo "Training configuration saved to: $OUTPUT_DIR/training_config.txt"