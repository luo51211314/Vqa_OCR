# LLaVA Fine-tuning for DocVQA and ChartVQA

This directory contains scripts for supervised LoRA fine-tuning of LLaVA models on DocVQA and ChartVQA datasets.

## Files

- `llava_finetune.py`: Main script for LLaVA model fine-tuning using LoRA
- `model_finetune.py`: General script for other model fine-tuning using LoRA
- `run_finetune.sh`: Example shell script for running fine-tuning

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- Peft
- LLaVA dependencies (from the models/llava directory)
- DocVQA/ChartVQA datasets

## Installation

```bash
cd /root/autodl-tmp/codes/Vqa_ocr
# Install LLaVA dependencies
pip install -e models/llava
# Install other requirements
pip install peft transformers accelerate
```

## Usage

### LLaVA Fine-tuning

For DocVQA:
```bash
python llava_finetune.py \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --data_path /path/to/docvqa/train.json \
    --image_folder /path/to/docvqa/images \
    --dataset_type docvqa \
    --output_dir ./checkpoints/llava-docvqa-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5
```

For ChartVQA:
```bash
python llava_finetune.py \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --data_path /path/to/chartvqa/train.json \
    --image_folder /path/to/chartvqa/images \
    --dataset_type chartvqa \
    --output_dir ./checkpoints/llava-chartvqa-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5
```

### General Model Fine-tuning

```bash
python model_finetune.py \
    --model_name_or_path /path/to/your/model \
    --data_path /path/to/dataset.json \
    --image_folder /path/to/images \
    --dataset_type docvqa \
    --output_dir ./checkpoints/model-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5
```

## Key Features

### LoRA Configuration
- **Rank (r)**: 64 (default) - controls the number of trainable parameters
- **Alpha**: 16 (default) - scaling factor for LoRA weights
- **Dropout**: 0.05 (default) - regularization
- **Target Modules**: Automatically detects linear layers excluding vision tower and projector

### Training Parameters
- **Batch Size**: Adjust based on GPU memory (8-16 recommended)
- **Learning Rate**: 2e-5 works well for most cases
- **Epochs**: 3-5 epochs typically sufficient
- **Sequence Length**: 2048 tokens maximum

### Memory Optimization
- Gradient Checkpointing: Enabled
- Mixed Precision: FP16 training
- Gradient Accumulation: Supported
- DataLoader Workers: 4 parallel processes

## Dataset Format

Your training data should be in JSON format with the following structure:

```json
[
  {
    "id": "unique_id",
    "image": "image_filename.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Question about the image <image>"
      },
      {
        "from": "gpt", 
        "value": "Answer to the question"
      }
    ]
  }
]
```

Or alternative format:

```json
[
  {
    "id": "unique_id",
    "image": "image_filename.jpg",
    "question": "Question about the image",
    "answer": "Answer to the question"
  }
]
```

## Output

The fine-tuned model will be saved in the specified output directory with:
- LoRA adapter weights
- Training configuration
- Tokenizer files
- Model configuration

## Notes

1. **Memory Requirements**: LoRA significantly reduces memory usage compared to full fine-tuning
2. **Original Parameters**: Base model parameters are frozen and not modified
3. **Compatibility**: Works with LLaVA v1.5 models and other compatible architectures
4. **Multi-GPU**: Supports distributed training with DeepSpeed

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Import Errors**: Make sure LLaVA is properly installed with `pip install -e models/llava`
3. **Dataset Issues**: Verify JSON format and image paths

## References

- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DocVQA Dataset](https://www.docvqa.org/)
- [ChartVQA Dataset](https://github.com/vis-nlp/ChartVQA)