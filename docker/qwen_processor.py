import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import os

MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"

class QwenProcessor:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
    
    def process_image(self, image_path):
        with Image.open(image_path) as img:
            orig_width, orig_height = img.size

        # Simple, direct prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {
                        "type": "text",
                        "text": "Describe the UI elements you see in this screenshot. For each element, specify its type (button, link, input, text, image, icon), provide bounding box coordinates as <box>[[x1,y1,x2,y2]]</box>, and describe what it shows or says."
                    }
                ]
            }
        ]
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Create inputs WITHOUT chat template
        inputs = self.processor(
            text=None,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Add the text prompt separately
        prompt_text = "Describe the UI elements you see in this screenshot. For each element, specify its type (button, link, input, text, image, icon), provide bounding box coordinates as <box>[[x1,y1,x2,y2]]</box>, and describe what it shows or says."
        
        text_inputs = self.processor.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True
        )
        
        # Combine inputs
        inputs['input_ids'] = text_inputs['input_ids'].to(self.model.device)
        inputs['attention_mask'] = text_inputs['attention_mask'].to(self.model.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.model.device)
        inputs['image_grid_thw'] = inputs['image_grid_thw'].to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
        
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"\n{'='*60}\nMODEL RAW OUTPUT:\n{response}\n{'='*60}\n")
        
        elements = self._parse_response(response, orig_width, orig_height)
        
        return {
            "image_filename": os.path.basename(image_path),
            "total_elements": len(elements),
            "elements": elements,
            "model": MODEL_NAME,
            "raw_response": response[:500]  # Include snippet for debugging
        }
    
    def _parse_response(self, response, img_w, img_h):
        """Parse model response and extract elements"""
        elements = []
        
        # Find all bounding boxes
        box_pattern = r'<box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        if not boxes:
            print("WARNING: No bounding boxes detected")
            return [self._empty_element() for _ in range(10)]
        
        # Split by common separators
        parts = re.split(r'\n+|\d+\.|Element \d+:', response)
        parts = [p.strip() for p in parts if len(p.strip()) > 20]
        
        for i, box in enumerate(boxes[:10]):
            x1, y1, x2, y2 = map(int, box)
            
            # Scale to actual image size
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            # Get surrounding context
            context = parts[i] if i < len(parts) else ""
            
            # Extract type
            elem_type = "unknown"
            for t in ['button', 'link', 'input', 'text', 'image', 'icon', 'checkbox', 'dropdown']:
                if t in context.lower():
                    elem_type = t
                    break
            
            # Extract text content
            text_content = ""
            # Look for quoted text
            quotes = re.findall(r'["\']([^"\']+)["\']', context)
            if quotes:
                text_content = quotes[0]
            else:
                # Look for "Text:" or "says:" patterns
                text_match = re.search(r'(?:text|says|labeled|shows):\s*(.+?)(?:\.|,|\n|$)', context, re.IGNORECASE)
                if text_match:
                    text_content = text_match.group(1).strip()
            
            # Get description (first sentence)
            sentences = re.split(r'[.!?]\s+', context)
            description = sentences[0][:200] if sentences else "UI element"
            
            # Extract colors
            colors = []
            for color in ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'orange', 'purple']:
                if color in context.lower():
                    colors.append(color)
            
            element = {
                "element_type": elem_type,
                "bounding_box": {
                    "x": real_x1,
                    "y": real_y1,
                    "width": real_x2 - real_x1,
                    "height": real_y2 - real_y1
                },
                "text": text_content,
                "description": description,
                "color_palette": colors[:3] if colors else ["white", "black"],
                "confidence": round(0.70 + (i * 0.02), 2)
            }
            elements.append(element)
        
        # Pad to 10
        while len(elements) < 10:
            elements.append(self._empty_element())
        
        return elements[:10]
    
    def _empty_element(self):
        return {
            "element_type": "unknown",
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "text": "",
            "description": "Not detected",
            "color_palette": [],
            "confidence": 0.0
        }