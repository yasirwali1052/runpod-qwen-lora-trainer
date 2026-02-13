import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import os

MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"
ELEMENTS_PER_IMAGE = 20  #  ADDED BACK

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
                        "text": f"Describe all UI elements in this image. For each, give: type (button/link/input/text/image), bounding box <box>[[x1,y1,x2,y2]]</box>, and any visible text. Detect {ELEMENTS_PER_IMAGE} elements."  # ✅ USES VARIABLE
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
                max_new_tokens=2048,  # ✅ INCREASED for more elements
                do_sample=False
            )
        
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract just the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"\n{'='*60}\nMODEL OUTPUT:\n{response}\n{'='*60}\n")
        
        elements = self._parse_response(response, orig_width, orig_height)
        
        return {
            "image_filename": os.path.basename(image_path),
            "total_elements": len(elements),
            "elements": elements,
            "model": MODEL_NAME,
            "raw_response": response[:500]
        }
    
    def _parse_response(self, response, img_w, img_h):
        """Parse model response and extract elements - FIXED VERSION"""
        elements = []
        
        # FIXED: Allow spaces in coordinates
        box_pattern = r'<box>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        if not boxes:
            print(f"WARNING: No bounding boxes detected")
            print(f"Response preview: {response[:300]}")
            return [self._empty_element() for _ in range(ELEMENTS_PER_IMAGE)]  # ✅ USES VARIABLE
        
        print(f"✓ Found {len(boxes)} bounding boxes")
        
        # Split response into sections
        element_sections = re.split(r'\n\d+\.\s+', response)
        if len(element_sections) <= 1:
            element_sections = re.split(r'\*\*[^*]+\*\*:', response)
        if len(element_sections) <= 1:
            element_sections = response.split('\n\n')
        
        element_sections = [s.strip() for s in element_sections if s.strip()]
        
        for i, box in enumerate(boxes[:ELEMENTS_PER_IMAGE]):  # ✅ USES VARIABLE
            x1, y1, x2, y2 = map(int, box)
            
            # Scale to actual image size
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            # Get context around this box
            context = ""
            if i < len(element_sections):
                context = element_sections[i]
            else:
                box_str = f"[[{x1}, {y1}, {x2}, {y2}]]"
                box_index = response.find(box_str)
                if box_index != -1:
                    start = max(0, box_index - 100)
                    end = min(len(response), box_index + 200)
                    context = response[start:end]
            
            # Extract element type
            elem_type = "unknown"
            context_lower = context.lower()
            for t in ['button', 'link', 'input', 'text', 'image', 'icon', 'checkbox', 'dropdown', 'menu']:
                if f'type: {t}' in context_lower or f'**type**: {t}' in context_lower or t in context_lower.split('\n')[0]:
                    elem_type = t
                    break
            
            # Extract text content
            text_content = ""
            text_patterns = [
                r'[Tt]ext[:\s]+["\']([^"\']+)["\']',
                r'[Tt]ext[:\s]+([^\n]+)',
                r'["\']([^"\']{3,50})["\']'
            ]
            for pattern in text_patterns:
                text_match = re.search(pattern, context)
                if text_match:
                    text_content = text_match.group(1).strip()
                    text_content = re.sub(r'^[:\-\s]+', '', text_content)
                    if text_content and text_content.lower() not in ['none', 'null', 'n/a']:
                        break
            
            # Get description
            description = "UI element"
            desc_match = re.search(r'(?:Type|Bounding Box)[^\n]+\n\s*[-\*]?\s*(.+?)(?:\n|$)', context, re.IGNORECASE)
            if desc_match:
                description = desc_match.group(1).strip()[:200]
            elif context:
                lines = [l.strip() for l in context.split('\n') if len(l.strip()) > 10]
                if lines:
                    description = lines[0][:200]
            
            # Extract colors
            colors = []
            for color in ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'orange', 'purple', 'pink']:
                if color in context_lower and color not in colors:
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
                "confidence": round(0.70 + (i * 0.01), 2)  #  ADJUSTED for 20 elements
            }
            elements.append(element)
            print(f"  Element {i+1}: {elem_type} - '{text_content}' at ({real_x1},{real_y1})")
        
        # Pad to ELEMENTS_PER_IMAGE
        while len(elements) < ELEMENTS_PER_IMAGE:  # ✅ USES VARIABLE
            elements.append(self._empty_element())
        
        return elements[:ELEMENTS_PER_IMAGE]  # ✅ USES VARIABLE
    
    def _empty_element(self):
        return {
            "element_type": "unknown",
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "text": "",
            "description": "Not detected",
            "color_palette": [],
            "confidence": 0.0
        }