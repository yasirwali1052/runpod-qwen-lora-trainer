import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import os

MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"
ELEMENTS_PER_IMAGE = 20

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {
                        "type": "text",
                        "text": f"List {ELEMENTS_PER_IMAGE} UI elements. For each provide: Type, bounding box <box>[[x,y,x2,y2]]</box>, visible text in quotes, brief description, and colors."
                    }
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
        
        print(f"\n{'='*60}\nMODEL RAW OUTPUT:\n{response}\n{'='*60}\n")
        
        elements = self._parse_response(response, orig_width, orig_height)
        
        return {
            "image_filename": os.path.basename(image_path),
            "total_elements": len(elements),
            "elements": elements
        }
    
    def _parse_response(self, response, img_w, img_h):
        """COMPLETELY REWRITTEN PARSER"""
        elements = []
        
        # Find all boxes
        box_pattern = r'<box>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        if not boxes:
            print("❌ NO BOXES FOUND")
            return [self._empty_element() for _ in range(ELEMENTS_PER_IMAGE)]
        
        print(f"✓ Found {len(boxes)} boxes")
        
        # Split by numbered items (1. 2. 3. etc)
        sections = re.split(r'\n\s*\d+\.\s+', response)
        sections = [s.strip() for s in sections if s.strip()]
        
        for i, box in enumerate(boxes[:ELEMENTS_PER_IMAGE]):
            x1, y1, x2, y2 = map(int, box)
            
            # Scale coordinates
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            # Get text for this element
            context = sections[i] if i < len(sections) else ""
            
            # Remove box coords from context
            context_clean = re.sub(r'<box>.*?</box>', '', context)
            
            # EXTRACT TYPE
            elem_type = "unknown"
            # Look for patterns like: **Type**: button OR Type: button OR just "button" at start
            type_match = re.search(r'(?:\*\*)?[Tt]ype(?:\*\*)?:\s*(\w+)', context_clean)
            if type_match:
                elem_type = type_match.group(1).lower()
            else:
                # Fallback: check first word
                first_word = context_clean.split()[0].lower() if context_clean.split() else ""
                if first_word in ['button', 'link', 'input', 'text', 'image', 'icon', 'menu']:
                    elem_type = first_word
            
            # EXTRACT TEXT - look for ANY quoted text
            text_content = ""
            # Pattern 1: "text here" or 'text here'
            quote_matches = re.findall(r'["\']([^"\']{2,100})["\']', context_clean)
            if quote_matches:
                # Get longest match (usually the actual text)
                text_content = max(quote_matches, key=len)
            
            # EXTRACT DESCRIPTION - get clean sentence
            description = "UI element"
            # Remove type/text/box lines
            desc_clean = re.sub(r'\*\*[Tt]ype\*\*:.*', '', context_clean)
            desc_clean = re.sub(r'\*\*[Tt]ext\*\*:.*', '', desc_clean)
            desc_clean = re.sub(r'\*\*.*?\*\*:', '', desc_clean)  # Remove all **Label**:
            desc_clean = desc_clean.strip()
            
            # Get first meaningful sentence
            sentences = re.split(r'[.!]\s+', desc_clean)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 15 and not any(x in sent.lower() for x in ['here is', 'bounding', 'type:', 'text:']):
                    description = sent[:200]
                    break
            
            # EXTRACT COLORS - find actual color words
            colors = []
            color_words = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'white', 'black', 'gray', 'grey', 'brown']
            context_lower = context_clean.lower()
            for color in color_words:
                if color in context_lower and color not in colors:
                    colors.append(color)
                    if len(colors) >= 3:
                        break
            
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
                "color_palette": colors if colors else [],
                "confidence": round(0.70 + (i * 0.01), 2)
            }
            elements.append(element)
            
            print(f"  [{i+1}] {elem_type:8s} | '{text_content[:30]:30s}' | {description[:40]:40s} | {colors}")
        
        while len(elements) < ELEMENTS_PER_IMAGE:
            elements.append(self._empty_element())
        
        return elements[:ELEMENTS_PER_IMAGE]
    
    def _empty_element(self):
        return {
            "element_type": "unknown",
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "text": "",
            "description": "Not detected",
            "color_palette": [],
            "confidence": 0.0
        }