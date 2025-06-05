import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from datasets import Dataset
import numpy as np

def load_urbansound_dataset(jsonl_path):
    """Load the UrbanSound dataset from jsonl file"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "image": item["image"],
                "text": item["text"],
                "label": item["label"]
            })
    return data

# Load the base model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

def prepare_training_data(dataset):
    """Prepare training data in the required format"""
    training_data = []
    for item in dataset:
        # Create a flattened structure for the dataset
        training_data.append({
            "image": item["image"],
            "text": item["text"],
            "label": item["label"],
            "user_content": json.dumps([
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["text"]}
            ]),
            "assistant_content": f"This is a spectrogram of {item['label']}."
        })
    return training_data

def preprocess_function(examples):
    """Preprocess the data for training"""
    texts = []
    image_inputs_list = []
    video_inputs_list = []
    
    # Handle both single example and batch of examples
    if not isinstance(examples, list):
        examples = [examples]
    
    for example in examples:
        # Reconstruct the messages structure
        messages = [
            {
                "role": "user",
                "content": json.loads(example["user_content"])
            },
            {
                "role": "assistant",
                "content": example["assistant_content"]
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
        
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs_list.append(image_inputs)
        video_inputs_list.append(video_inputs)
    
    # Process inputs without videos since we don't have any
    inputs = processor(
        text=texts,
        images=image_inputs_list,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs

def train_model(dataset_path):
    # Load and prepare dataset
    raw_data = load_urbansound_dataset(dataset_path)
    training_data = prepare_training_data(raw_data)
    
    # Create dataset from the flattened structure
    dataset = Dataset.from_list(training_data)
    
    # Get unique labels from the dataset
    label_names = list(set(item["label"] for item in raw_data))
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./qwen-vl-lora-urbansound",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        remove_unused_columns=False,
        label_names=label_names,  # Add label names to training arguments
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: preprocess_function(data),
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model()
    
    return model

if __name__ == "__main__":
    # Train the model using the UrbanSound dataset
    dataset_path = "urbansound_dataset.jsonl"
    trained_model = train_model(dataset_path) 