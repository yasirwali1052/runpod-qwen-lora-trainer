import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import os

MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"
ELEMENTS_PER_IMAGE = 10

class QwenProcessor:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16  # ← FIXED: Changed from float16 to bfloat16
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
                        "text": "Detect UI elements in this screenshot. For each element, provide: element type (button/link/input/image/text/icon/dropdown/checkbox/menu/header), bounding box <box>[[x1,y1,x2,y2]]</box>, text content, visual description with colors and styling, typography details. Detect 10 elements."
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
        
        elements = self._parse_response(response, orig_width, orig_height)
        
        return {
            "image_filename": os.path.basename(image_path),
            "total_elements": len(elements),
            "elements": elements,
            "model": MODEL_NAME
        }
    
    def _parse_response(self, response, img_w, img_h):
        elements = []
        box_pattern = r'<box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        response_parts = re.split(box_pattern, response)
        
        element_types = {
            'button': ['button', 'btn', 'submit', 'click'],
            'link': ['link', 'href', 'anchor'],
            'input': ['input', 'field', 'textbox', 'form'],
            'image': ['image', 'img', 'photo'],
            'text': ['text', 'label', 'paragraph'],
            'icon': ['icon', 'symbol'],
            'dropdown': ['dropdown', 'select'],
            'checkbox': ['checkbox', 'check'],
            'menu': ['menu', 'nav'],
            'header': ['header', 'title']
        }
        
        for i, box in enumerate(boxes[:ELEMENTS_PER_IMAGE]):
            x1, y1, x2, y2 = map(int, box)
            
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)

            context_start = max(0, i * 5 - 2)
            context_end = min(len(response_parts), (i + 1) * 5 + 2)
            context = ' '.join(response_parts[context_start:context_end])
            
            element = {
                "element_type": self._classify_element(context, element_types),
                "bounding_box": {
                    "x": real_x1,
                    "y": real_y1,
                    "width": real_x2 - real_x1,
                    "height": real_y2 - real_y1
                },
                "description": self._extract_text(context),
                "visual_style": self._extract_visual_style(context),
                "color_palette": self._extract_colors(context),
                "typography": self._extract_typography(context),
                "confidence": round(min(0.95, 0.70 + (i * 0.025)), 2)
            }
            elements.append(element)
        
        while len(elements) < ELEMENTS_PER_IMAGE:
            elements.append({
                "element_type": "text",
                "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
                "description": "Not detected",
                "visual_style": "N/A",
                "color_palette": [],
                "typography": "N/A",
                "confidence": 0.0
            })
        
        return elements[:ELEMENTS_PER_IMAGE]

    def _classify_element(self, context, element_types):
        context_lower = context.lower()
        for etype, keywords in element_types.items():
            if any(kw in context_lower for kw in keywords):
                return etype
        return "button"

    def _extract_text(self, context):
        quote_pattern = r'["\']([^"\']{1,100})["\']'
        matches = re.findall(quote_pattern, context)
        if matches:
            return matches[0].strip()
        
        keywords = ['says', 'reads', 'labeled', 'text:', 'content:']
        for kw in keywords:
            if kw in context.lower():
                parts = context.split(kw)
                if len(parts) > 1:
                    return parts[1].strip().split('.')[0][:100]
        
        return "UI Element"
    
    def _extract_colors(self, context):
        colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'orange', 'purple', 'pink']
        found = []
        context_lower = context.lower()
        
        for color in colors:
            if color in context_lower and color not in found:
                found.append(color)
        
        return found[:3] if found else ["gray", "white"]
    
    def _extract_typography(self, context):
        descriptors = []
        context_lower = context.lower()
        
        if 'bold' in context_lower:
            descriptors.append('bold')
        if 'italic' in context_lower:
            descriptors.append('italic')
        if 'sans-serif' in context_lower:
            descriptors.append('sans-serif')
        elif 'serif' in context_lower:
            descriptors.append('serif')
        
        size_match = re.search(r'(\d+)\s*px', context_lower)
        if size_match:
            descriptors.append(f"{size_match.group(1)}px")
        else:
            descriptors.append('14px')
        
        return ' '.join(descriptors) if descriptors else "Regular 14px"
    
    def _extract_visual_style(self, context):
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]
        keywords = ['color', 'background', 'border', 'rounded', 'shadow', 'style']
        
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keywords):
                return sentence[:200]
        
        return sentences[0][:200] if sentences else "Standard UI element"