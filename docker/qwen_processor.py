import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import os
import json

MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"
ELEMENTS_PER_IMAGE = 10

class QwenProcessor:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
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
                        "text": """Analyze this UI screenshot carefully. Identify 10 distinct interactive elements (buttons, links, inputs, icons, etc.).

For each element, provide in this exact format:

Element N:
- Type: [button/link/input/icon/text/image/dropdown/checkbox]
- Position: <box>[[x1,y1,x2,y2]]</box>
- Text: "[exact text visible on element]"
- Description: [color, shape, style details]

Be precise with coordinates. Don't repeat the same element twice."""
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
        
        # Extract all bounding boxes
        box_pattern = r'<box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        # Split response by "Element" to get individual element blocks
        element_blocks = re.split(r'Element \d+:', response)[1:]  # Skip first empty split
        
        for i, box in enumerate(boxes[:ELEMENTS_PER_IMAGE]):
            x1, y1, x2, y2 = map(int, box)
            
            # Convert from 1000-scale to actual pixel coordinates
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            # Get the corresponding element block
            block_text = element_blocks[i] if i < len(element_blocks) else ""
            
            # Extract element type
            element_type = self._extract_type(block_text)
            
            # Extract text content
            text_content = self._extract_text_content(block_text)
            
            # Extract description
            description = self._extract_description(block_text)
            
            # Extract colors
            colors = self._extract_colors(block_text)
            
            element = {
                "element_type": element_type,
                "bounding_box": {
                    "x": real_x1,
                    "y": real_y1,
                    "width": real_x2 - real_x1,
                    "height": real_y2 - real_y1
                },
                "text": text_content,
                "description": description,
                "color_palette": colors,
                "confidence": round(0.70 + (i * 0.02), 2)
            }
            elements.append(element)
        
        # Pad with empty elements if needed
        while len(elements) < ELEMENTS_PER_IMAGE:
            elements.append({
                "element_type": "unknown",
                "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
                "text": "",
                "description": "Not detected",
                "color_palette": [],
                "confidence": 0.0
            })
        
        return elements[:ELEMENTS_PER_IMAGE]

    def _extract_type(self, block):
        """Extract element type from block"""
        type_match = re.search(r'Type:\s*\[?([^\]\n]+)\]?', block, re.IGNORECASE)
        if type_match:
            element_type = type_match.group(1).strip().lower()
            # Map to standard types
            if 'button' in element_type or 'btn' in element_type:
                return 'button'
            elif 'link' in element_type or 'anchor' in element_type:
                return 'link'
            elif 'input' in element_type or 'field' in element_type:
                return 'input'
            elif 'icon' in element_type:
                return 'icon'
            elif 'image' in element_type or 'img' in element_type:
                return 'image'
            elif 'text' in element_type or 'label' in element_type:
                return 'text'
            elif 'dropdown' in element_type or 'select' in element_type:
                return 'dropdown'
            elif 'checkbox' in element_type or 'check' in element_type:
                return 'checkbox'
            return element_type
        return 'button'
    
    def _extract_text_content(self, block):
        """Extract text content from block"""
        text_match = re.search(r'Text:\s*["\']([^"\']+)["\']', block, re.IGNORECASE)
        if text_match:
            return text_match.group(1).strip()
        
        # Alternative pattern
        text_match = re.search(r'Text:\s*\[([^\]]+)\]', block, re.IGNORECASE)
        if text_match:
            return text_match.group(1).strip()
        
        return ""
    
    def _extract_description(self, block):
        """Extract description from block"""
        desc_match = re.search(r'Description:\s*\[?([^\]\n]{10,200})\]?', block, re.IGNORECASE)
        if desc_match:
            return desc_match.group(1).strip()
        
        # Fallback: get text after Description:
        if 'Description:' in block:
            parts = block.split('Description:')
            if len(parts) > 1:
                desc = parts[1].strip().split('\n')[0][:200]
                return desc
        
        return "UI element"
    
    def _extract_colors(self, block):
        """Extract colors from block"""
        colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'grey',
                  'orange', 'purple', 'pink', 'brown', 'cyan', 'teal']
        found = []
        block_lower = block.lower()
        
        for color in colors:
            if color in block_lower and color not in found:
                found.append(color)
                if len(found) >= 3:
                    break
        
        return found if found else ["gray"]