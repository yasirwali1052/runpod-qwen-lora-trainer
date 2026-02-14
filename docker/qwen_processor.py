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
                        "text": f"List {ELEMENTS_PER_IMAGE} UI elements. For each provide: Type, bounding box <box>[[x,y,x2,y2]]</box>, visible text in quotes, and brief description."
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
        
        print(f"\n{'='*60}\nMODEL OUTPUT:\n{response}\n{'='*60}\n")
        
        elements = self._parse_response(response, orig_width, orig_height)
        
        return {
            "image_filename": os.path.basename(image_path),
            "total_elements": len(elements),
            "elements": elements
        }
    
    def _normalize_element_type(self, raw_type):
        """Map model output to standard UI element types"""
        type_mapping = {
            'navigation': 'menu',
            'nav': 'menu',
            'navbar': 'menu',
            'search': 'input',
            'searchbar': 'input',
            'textfield': 'input',
            'textarea': 'input',
            'textbox': 'input',
            'hyperlink': 'link',
            'anchor': 'link',
            'img': 'image',
            'picture': 'image',
            'photo': 'image',
            'logo': 'image',
            'icon': 'image',
            'btn': 'button',
            'cta': 'button',
            'tab': 'button',
            'toggle': 'checkbox',
            'switch': 'checkbox',
            'select': 'dropdown',
            'picker': 'dropdown',
            'selector': 'dropdown'
        }
        
        raw_type = raw_type.lower().strip()
        return type_mapping.get(raw_type, raw_type)
    
    def _parse_response(self, response, img_w, img_h):
        """Parser WITHOUT color extraction"""
        elements = []
        
        box_pattern = r'<box>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        if not boxes:
            print("❌ NO BOXES FOUND")
            return [self._empty_element() for _ in range(ELEMENTS_PER_IMAGE)]
        
        print(f"✓ Found {len(boxes)} boxes")
        
        sections = re.split(r'\n\s*\d+\.\s+', response)
        sections = [s.strip() for s in sections if s.strip()]
        
        for i, box in enumerate(boxes[:ELEMENTS_PER_IMAGE]):
            x1, y1, x2, y2 = map(int, box)
            
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            context = sections[i] if i < len(sections) else ""
            
            # CLEAN context
            context_clean = re.sub(r'<box>.*?</box>', '', context)
            context_clean = re.sub(r'(?:\*\*)?[Tt]ype(?:\*\*)?:.*', '', context_clean)
            context_clean = re.sub(r'(?:\*\*)?[Tt]ext(?:\*\*)?:.*', '', context_clean)
            context_clean = re.sub(r'\*\*[Bb]ounding.*?\*\*:.*', '', context_clean)
            context_clean = re.sub(r'Purpose:.*', '', context_clean)
            context_clean = context_clean.strip()
            
            # EXTRACT TYPE
            elem_type = "unknown"
            type_match = re.search(r'(?:\*\*)?[Tt]ype(?:\*\*)?:\s*(\w+)', context)
            if type_match:
                raw_type = type_match.group(1).lower()
                elem_type = self._normalize_element_type(raw_type)
            else:
                first_word = context.split()[0].lower() if context.split() else ""
                elem_type = self._normalize_element_type(first_word)
            
            # EXTRACT TEXT
            text_content = ""
            quote_matches = re.findall(r'["\']([^"\']{2,100})["\']', context)
            if quote_matches:
                text_content = max(quote_matches, key=len)
            
            # EXTRACT DESCRIPTION
            description = "UI element"
            if context_clean and len(context_clean) > 10:
                sentences = re.split(r'[.!]\s+', context_clean)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 20 and not any(x in sent.lower() for x in ['here is', 'ui element', 'button', 'link']):
                        description = sent[:200]
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
                "confidence": round(0.70 + (i * 0.01), 2)
            }
            elements.append(element)
            
            print(f"  [{i+1}] {elem_type:8s} | '{text_content[:30]:30s}' | {description[:40]:40s}")
        
        while len(elements) < ELEMENTS_PER_IMAGE:
            elements.append(self._empty_element())
        
        return elements[:ELEMENTS_PER_IMAGE]
    
    def _empty_element(self):
        return {
            "element_type": "unknown",
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "text": "",
            "description": "Not detected",
            "confidence": 0.0
        }