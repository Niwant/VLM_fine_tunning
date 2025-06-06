import torch
import json
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from datasets import Dataset
import numpy as np
from PIL import Image

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
    return data[:20]

# Load the base model
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
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
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B", trust_remote_code=True)

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

def preprocess_function(example):
    """Preprocess a single example for training"""
    # Load and process the image
    image = Image.open(example["image"]).convert('RGB')
    
    # Create the conversation format expected by Qwen2.5-Omni
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        },
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": example["text"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"This is a spectrogram of {example['label']}."}
            ]
        }
    ]
    
    # Process inputs according to Qwen2.5-Omni format
    model_inputs = processor(
        conversations=conversation,
        return_tensors="pt",
        text_only=False,
        truncation=True,
        max_length=512
    )
    
    # Remove batch dimension as we'll add it back in the data collator
    for key in model_inputs:
        if isinstance(model_inputs[key], torch.Tensor) and model_inputs[key].dim() > 0:
            model_inputs[key] = model_inputs[key].squeeze(0)
    
    return model_inputs

def data_collator(examples):
    """Custom data collator for batching examples"""
    batch = {}
    
    # Get all keys from examples
    keys = set(key for example in examples for key in example.keys())
    
    for key in keys:
        # Skip keys that might not be in all examples
        if not all(key in example for example in examples):
            continue
        
        # Handle different types of data
        if key == "pixel_values" or (isinstance(examples[0][key], torch.Tensor) and examples[0][key].dim() >= 3):
            # For images or other 3D+ tensors, stack them
            batch[key] = torch.stack([example[key] for example in examples])
        elif isinstance(examples[0][key], torch.Tensor):
            # For sequence data, pad it
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [example[key] for example in examples],
                batch_first=True,
                padding_value=processor.tokenizer.pad_token_id if key != "labels" else -100
            )
        else:
            # For other types, just put them in a list
            batch[key] = [example[key] for example in examples]
    
    return batch

def train_model(dataset_path):
    # Load and prepare dataset
    raw_data = load_urbansound_dataset(dataset_path)
    training_data = prepare_training_data(raw_data)
    
    # Create dataset from the flattened structure
    dataset = Dataset.from_list(training_data)
    
    # Map the preprocess function to the dataset
    processed_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        num_proc=1,  # Reduced from 4 to help with debugging
        desc="Processing dataset",
    )
    
    # Print sample to verify data structure
    print("Sample processed example:", next(iter(processed_dataset)))
    
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
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model()
    
    return model

def test_model(model_path, test_image_path, test_text=""):
    """Test the trained model on a spectrogram image"""
    # Load the trained model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    
    # Load and process the image
    image = Image.open(test_image_path).convert('RGB')
    
    # Define the possible labels from the UrbanSound dataset
    possible_labels = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", 
        "drilling", "engine_idling", "gun_shot", "jackhammer", 
        "siren", "street_music"
    ]
    
    # Create the conversation format expected by Qwen2.5-Omni
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"What type of sound is this spectrogram showing? Choose from these categories: {', '.join(possible_labels)}"}
            ]
        }
    ]
    
    # Apply chat template and ensure it's a string
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    if not isinstance(text, str):
        text = str(text)
    
    # Process the input
    try:
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(model.device)
    except Exception as e:
        print(f"Error processing inputs: {e}")
        print(f"Text type: {type(text)}")
        print(f"Text content: {text}")
        raise
    
    # Prepare the model inputs in the correct format
    model_inputs = {
        "pixel_values": inputs["pixel_values"],
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.2,
            return_audio=False  # We only need text output for this task
        )
    
    # Decode and print the response
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print("\nModel's response:")
    print(response)
    
    return response

if __name__ == "__main__":
    # Train the model using the UrbanSound dataset
    dataset_path = "urbansound_dataset.jsonl"
    trained_model = train_model(dataset_path)
    test = {"image": "../UrbanSound-Spectrogram/fold1/98223-7-2-0.png", "text": "What type of sound is this spectrogram showing?", "label": "jackhammer"}

    # Test the model on a sample spectrogram
    test_image_path = test["image"]  # Replace with actual test image path
    test_model("./qwen-vl-lora-urbansound", test_image_path)