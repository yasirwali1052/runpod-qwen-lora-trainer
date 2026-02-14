import os
import json
import torch
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
APPROVED_DATA_DIR = "data/approved"
WEIGHTS_OUTPUT_DIR = "weights"

class LoRATrainer:
    def __init__(self):
        print("Loading base model (this may take 5-10 minutes)...")
        
        # Load model in 8-bit for CPU compatibility
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        print("Preparing model for LoRA training...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("\nTrainable parameters:")
        self.model.print_trainable_parameters()
    
    def load_approved_data(self):
        """Load image and JSON pairs from approved folder"""
        data_pairs = []
        
        if not os.path.exists(APPROVED_DATA_DIR):
            print(f"ERROR: {APPROVED_DATA_DIR} folder not found")
            return []
        
        image_files = [f for f in os.listdir(APPROVED_DATA_DIR) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"ERROR: No images found in {APPROVED_DATA_DIR}")
            return []
        
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            json_file = base_name + ".json"
            
            img_path = os.path.join(APPROVED_DATA_DIR, img_file)
            json_path = os.path.join(APPROVED_DATA_DIR, json_file)
            
            if not os.path.exists(json_path):
                print(f"WARNING: Missing JSON for {img_file}, skipping")
                continue
            
            try:
                with open(json_path, 'r') as f:
                    labels = json.load(f)
                
                data_pairs.append({
                    "image_path": img_path,
                    "labels": labels
                })
                
            except Exception as e:
                print(f"ERROR loading {json_file}: {e}")
                continue
        
        print(f"Loaded {len(data_pairs)} approved image-label pairs")
        return data_pairs
    
    def format_training_example(self, pair):
        """Convert single pair to training format"""
        img_path = pair["image_path"]
        labels = pair["labels"]
        
        # Build ground truth text from JSON
        elements_text = []
        for i, elem in enumerate(labels.get("elements", []), 1):
            bbox = elem["bounding_box"]
            img_h = labels["image_size"]["height"]
            img_w = labels["image_size"]["width"]
            
            y1 = int(bbox["y"] * 1000 / img_h)
            x1 = int(bbox["x"] * 1000 / img_w)
            y2 = int((bbox["y"] + bbox["height"]) * 1000 / img_h)
            x2 = int((bbox["x"] + bbox["width"]) * 1000 / img_w)
            
            elements_text.append(
                f"{i}. Type: {elem['element_type']}\n"
                f"   Bounding Box: <box>[[{y1},{x1},{y2},{x2}]]</box>\n"
                f"   Text: {elem['text']}\n"
                f"   Description: {elem['description']}"
            )
        
        ground_truth = "\n\n".join(elements_text)
        
        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": "Analyze this UI screenshot and identify UI elements with their bounding boxes."}
                ]
            },
            {
                "role": "assistant",
                "content": ground_truth
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": text, "image": img_path}
    
    def train(self, epochs=1, learning_rate=2e-4, output_name="lora_v1"):
        """Train LoRA on approved data"""
        
        # Load and format data
        data_pairs = self.load_approved_data()
        if not data_pairs:
            print("No training data available. Exiting.")
            return
        
        print("\nFormatting training data...")
        formatted_data = [self.format_training_example(pair) for pair in data_pairs]
        dataset = Dataset.from_list(formatted_data)
        
        print(f"\nTraining on {len(dataset)} examples for {epochs} epoch(s)")
        print("WARNING: Training on CPU will be VERY SLOW")
        print("Estimated time: 1-3 hours per epoch depending on data size\n")
        
        # Training arguments
        output_path = os.path.join(WEIGHTS_OUTPUT_DIR, output_name)
        os.makedirs(output_path, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=False,
            optim="adamw_torch",
            warmup_steps=2,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            report_to="none"
        )
        
        # Simple data collator
        def collate_fn(examples):
            texts = [ex["text"] for ex in examples]
            images = [Image.open(ex["image"]) for ex in examples]
            
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt"
            )
            
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_fn,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save final model
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        
        print(f"\nTraining complete!")
        print(f"LoRA weights saved to: {output_path}")


def main():
    print("="*60)
    print("QWEN2-VL LoRA FINE-TUNING (CPU MODE)")
    print("="*60)
    
    # Check approved data folder
    if not os.path.exists(APPROVED_DATA_DIR):
        print(f"\nERROR: Create '{APPROVED_DATA_DIR}' folder first")
        print("Copy approved image+JSON pairs there before training")
        return
    
    # Create weights folder
    os.makedirs(WEIGHTS_OUTPUT_DIR, exist_ok=True)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = LoRATrainer()
    
    # Get training parameters
    print("\n" + "="*60)
    epochs = int(input("Number of epochs [1]: ") or "1")
    lr = float(input("Learning rate [2e-4]: ") or "2e-4")
    output_name = input("Output name [lora_v1]: ") or "lora_v1"
    
    # Train
    trainer.train(epochs=epochs, learning_rate=lr, output_name=output_name)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
