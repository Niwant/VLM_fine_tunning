# VLM Fine-tuning with LoRA

This repository contains code for fine-tuning the Qwen2.5-VL model using LoRA (Low-Rank Adaptation) on the UrbanSound dataset. The project focuses on training the model to classify spectrograms of urban sounds.

## Overview

The project uses the Qwen2.5-VL-3B-Instruct model, a powerful vision-language model, and fine-tunes it using LoRA to efficiently adapt it for spectrogram classification tasks. The model is trained to identify various urban sounds from their spectrogram representations.

## Features

- Fine-tuning of Qwen2.5-VL model using LoRA
- Support for spectrogram image processing
- Memory-efficient training with gradient accumulation
- Automatic mixed precision training (fp16)
- Checkpoint saving and model persistence

## Requirements

```bash
pip install torch transformers peft datasets
```

## Dataset

The project uses the UrbanSound dataset, which contains spectrograms of various urban sounds. The dataset is provided in JSONL format with the following structure:

```json
{
    "image": "path/to/spectrogram.png",
    "text": "Classify the sound in this spectrogram.",
    "label": "sound_class"
}
```

## Model Architecture

- Base Model: Qwen2.5-VL-3B-Instruct
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- LoRA Configuration:
  - Rank: 8
  - Alpha: 16
  - Target Modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05

## Training Configuration

- Number of Epochs: 3
- Batch Size: 1
- Gradient Accumulation Steps: 4
- Learning Rate: 2e-4
- Mixed Precision: fp16
- Warmup Ratio: 0.1

## Usage

1. Prepare your dataset in JSONL format
2. Run the training script:

```bash
python lora_training.py
```

The model will be saved in the `./qwen-vl-lora-urbansound` directory.

## Training Process

The training process involves:
1. Loading and preprocessing the spectrogram images
2. Converting the data into the required format for the model
3. Fine-tuning the model using LoRA
4. Saving checkpoints after each epoch

## Output

The fine-tuned model will be saved with the following structure:
```
qwen-vl-lora-urbansound/
├── adapter_config.json
├── adapter_model.bin
└── training_args.bin
```

## Memory Requirements

- GPU with at least 16GB VRAM recommended
- The training uses gradient accumulation to handle memory constraints

## License

[Add your license information here]

## Acknowledgments

- Qwen2.5-VL model from Alibaba Cloud
- UrbanSound dataset
- Hugging Face Transformers and PEFT libraries