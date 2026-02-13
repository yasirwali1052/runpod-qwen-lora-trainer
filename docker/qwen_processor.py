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
                        "text": f"Describe all UI elements in this image. For each, give: type (button/link/input/text/image), bounding box <box>[[x1,y1,x2,y2]]</box>, and any visible text. Detect {ELEMENTS_PER_IMAGE} elements."
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
        
        # Extract just the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"\n{'='*60}\nMODEL OUTPUT:\n{response}\n{'='*60}\n")
        
        elements = self._parse_response(response, orig_width, orig_height)
        
        # REMOVED: raw_response and model from output
        return {
            "image_filename": os.path.basename(image_path),
            "total_elements": len(elements),
            "elements": elements
        }
    
    def _parse_response(self, response, img_w, img_h):
        """Parse model response and extract elements - CLEANED VERSION"""
        elements = []
        
        # Allow spaces in coordinates
        box_pattern = r'<box>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        if not boxes:
            print(f"WARNING: No bounding boxes detected")
            return [self._empty_element() for _ in range(ELEMENTS_PER_IMAGE)]
        
        print(f"✓ Found {len(boxes)} bounding boxes")
        
        # Split response into numbered sections
        element_sections = re.split(r'\n\d+\.\s+', response)
        element_sections = [s.strip() for s in element_sections if s.strip()]
        
        for i, box in enumerate(boxes[:ELEMENTS_PER_IMAGE]):
            x1, y1, x2, y2 = map(int, box)
            
            # Scale to actual image size
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            # Get context for this element
            context = ""
            if i < len(element_sections):
                context = element_sections[i]
            else:
                # Fallback: find text near this box
                box_str = f"[[{x1}, {y1}, {x2}, {y2}]]"
                box_index = response.find(box_str)
                if box_index != -1:
                    start = max(0, box_index - 100)
                    end = min(len(response), box_index + 300)
                    context = response[start:end]
            
            # CLEAN: Remove bounding box from context
            context = re.sub(r'<box>\[\[[\d\s,]+\]\]</box>', '', context)
            # CLEAN: Remove "Bounding Box:" lines
            context = re.sub(r'\*\*Bounding Box\*\*:[^\n]*', '', context)
            # CLEAN: Remove intro text from first element
            context = re.sub(r'Here is (?:the )?description.*?:', '', context, flags=re.IGNORECASE)
            
            # Extract element type
            elem_type = "unknown"
            context_lower = context.lower()
            type_patterns = [
                r'\*\*Type\*\*:\s*(\w+)',
                r'Type:\s*(\w+)',
                r'-\s*Type:\s*(\w+)'
            ]
            for pattern in type_patterns:
                type_match = re.search(pattern, context, re.IGNORECASE)
                if type_match:
                    elem_type = type_match.group(1).lower()
                    break
            
            # Fallback: detect type from keywords
            if elem_type == "unknown":
                for t in ['button', 'link', 'input', 'text', 'image', 'icon', 'checkbox', 'dropdown', 'menu']:
                    if f'**{t}' in context_lower or f'type: {t}' in context_lower:
                        elem_type = t
                        break
            
            # Extract text content
            text_content = ""
            text_patterns = [
                r'\*\*Text\*\*:\s*["\']([^"\']+)["\']',
                r'Text:\s*["\']([^"\']+)["\']',
                r'["\']([^"\']{3,80})["\']'  # Any quoted text
            ]
            for pattern in text_patterns:
                text_match = re.search(pattern, context)
                if text_match:
                    text_content = text_match.group(1).strip()
                    # Skip generic/intro text
                    if not any(skip in text_content.lower() for skip in ['here is', 'description', 'bounding box', 'type:']):
                        break
                    else:
                        text_content = ""
            
            # Get clean description
            description = "UI element"
            # Remove type and text lines, keep actual description
            desc_lines = context.split('\n')
            for line in desc_lines:
                line = line.strip()
                # Skip lines with Type, Text, Bounding Box
                if any(skip in line for skip in ['**Type**', 'Type:', '**Text**', 'Text:', '**Bounding', 'Bounding Box']):
                    continue
                # Skip empty or very short lines
                if len(line) < 10:
                    continue
                # Skip intro lines
                if any(skip in line.lower() for skip in ['here is', 'description of', 'ui element']):
                    continue
                # Found a good description
                description = line[:200]
                break
            
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
                "confidence": round(0.70 + (i * 0.01), 2)
            }
            elements.append(element)
            print(f"  Element {i+1}: {elem_type} - '{text_content}' - {description[:50]}")
        
        # Pad to ELEMENTS_PER_IMAGE
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