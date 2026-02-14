import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import os
import json

MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"
ELEMENTS_PER_IMAGE = 15

# VALID ELEMENT TYPES
VALID_TYPES = {'button', 'link', 'input', 'text', 'image', 'icon', 'checkbox', 'dropdown', 'menu'}

class QwenProcessor:
    def __init__(self):
        print(f"Loading model: {MODEL_NAME}...")
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
        if not os.path.exists(image_path):
            return {"error": "File not found"}

        with Image.open(image_path) as img:
            orig_width, orig_height = img.size

        # Refined prompt to force a more consistent structure
        prompt_text = f"""Analyze this UI screenshot. Identify exactly {ELEMENTS_PER_IMAGE} UI elements.
For each element, provide:
1. Type: (choose from: button, link, input, text, image, icon, checkbox, dropdown, menu)
2. Bounding Box: <box>[[y1,x1,y2,x2]]</box> (0-1000 scale)
3. Text: The visible label or text
4. Description: Purpose of the element

List them numbered 1 to {ELEMENTS_PER_IMAGE}."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False
            )
        
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"\n{'='*30} RAW MODEL OUTPUT {'='*30}\n{response}\n{'='*78}\n")
        
        elements = self._parse_response(response, orig_width, orig_height)
        
        return {
            "image_filename": os.path.basename(image_path),
            "image_size": {"width": orig_width, "height": orig_height},
            "total_elements": len(elements),
            "elements": elements
        }
    
    def _validate_type(self, raw_type):
        raw_type = raw_type.lower().strip()
        raw_type = re.sub(r'[^a-z]', '', raw_type) # Clean non-alpha
        
        type_mapping = {
            'btn': 'button', 'hyperlink': 'link', 'anchor': 'link',
            'textfield': 'input', 'textbox': 'input', 'search': 'input',
            'img': 'image', 'pic': 'image', 'nav': 'menu', 'navbar': 'menu',
            'select': 'dropdown', 'toggle': 'checkbox', 'switch': 'checkbox'
        }
        
        if raw_type in type_mapping: return type_mapping[raw_type]
        if raw_type in VALID_TYPES: return raw_type
        
        for vt in VALID_TYPES:
            if vt in raw_type: return vt
            
        return 'button'
    
    def _parse_response(self, response, img_w, img_h):
        elements = []
        
        # Split into numbered sections (1. ..., 2. ...)
        sections = re.split(r'\n\s*\d+\.\s+', "\n" + response)
        sections = [s.strip() for s in sections if s.strip()]
        
        for i in range(ELEMENTS_PER_IMAGE):
            if i >= len(sections):
                elements.append(self._empty_element())
                continue
                
            context = sections[i]
            
            # 1. Extract Bounding Box - Qwen uses [y1, x1, y2, x2]
            box_match = re.search(r'<box>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]</box>', context)
            if not box_match:
                elements.append(self._empty_element())
                continue
            
            y1, x1, y2, x2 = map(int, box_match.groups())
            
            # Scale coordinates to actual pixel values
            rx1 = int(x1 * img_w / 1000)
            ry1 = int(y1 * img_h / 1000)
            rx2 = int(x2 * img_w / 1000)
            ry2 = int(y2 * img_h / 1000)

            # 2. Extract Text - Looking for <text>Tags</text> or Text: "Value"
            text_val = ""
            text_patterns = [
                r'<text>(.*?)</text>',
                r'[Tt]ext\s*[:=]\s*["\']?(.*?)["\']?(?:\n|$|,)',
                r'["\']([^"\']{2,30})["\']' 
            ]
            for pat in text_patterns:
                t_match = re.search(pat, context)
                if t_match:
                    text_val = t_match.group(1).strip()
                    break

            # 3. Extract Type
            e_type = 'button'
            type_match = re.search(r'[Tt]ype\s*[:=]\s*(\w+)', context)
            if type_match:
                e_type = self._validate_type(type_match.group(1))
            else:
                # Fallback check
                for vt in VALID_TYPES:
                    if vt in context.lower()[:50]:
                        e_type = vt
                        break

            # 4. Clean Description
            desc = re.sub(r'<.*?>', '', context) # Strip tags
            desc = re.sub(r'[Bb]ox:?\s*\[\[.*?\]\]', '', desc) # Strip box string
            desc = desc.replace('\n', ' ').strip()
            desc = ' '.join(desc.split()) # Normalize whitespace
            if len(desc) < 5: desc = f"A {e_type} element"

            elements.append({
                "element_type": e_type,
                "bounding_box": {"x": rx1, "y": ry1, "width": abs(rx2 - rx1), "height": abs(ry2 - ry1)},
                "text": text_val,
                "description": desc[:150],
                "confidence": round(0.70 + (i * 0.01), 2)
            })
            
        return elements

    def _empty_element(self):
        return {
            "element_type": "unknown",
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "text": "",
            "description": "Not detected",
            "confidence": 0.0
        }

# Example Usage
if __name__ == "__main__":
    processor = QwenProcessor()
    result = processor.process_image("your_screenshot.png")
    print(json.dumps(result, indent=2))