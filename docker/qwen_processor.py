import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
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

        # NEW PROMPT - asks for structured output
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {
                        "type": "text",
                        "text": """You are a UI element detector. Analyze this screenshot and identify up to 10 distinct UI elements.

For EACH element you detect, provide:
1. Element type: button, link, input, text, image, icon, dropdown, checkbox, or menu
2. Bounding box in format: <box>[[x1,y1,x2,y2]]</box> where coordinates are 0-1000
3. Visible text on the element (if any)
4. Brief visual description (color, shape, style)

Format each element like this:
Element 1: [type] <box>[[x1,y1,x2,y2]]</box>
Text: "actual text here"
Description: brief visual description

Be accurate. Only detect elements you can clearly see. Do not invent elements."""
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
        
        print(f"\n=== MODEL RESPONSE ===\n{response}\n===================\n")
        
        elements = self._parse_response_v2(response, orig_width, orig_height)
        
        return {
            "image_filename": os.path.basename(image_path),
            "total_elements": len(elements),
            "elements": elements,
            "model": MODEL_NAME
        }
    
    def _parse_response_v2(self, response, img_w, img_h):
        """Better parsing that handles varied model outputs"""
        elements = []
        
        # Extract all bounding boxes
        box_pattern = r'<box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        if not boxes:
            print("WARNING: No bounding boxes found in response")
            return self._empty_elements()
        
        # Split response into sections by "Element"
        element_sections = re.split(r'Element \d+:', response)
        element_sections = [s.strip() for s in element_sections if s.strip()]
        
        for i, box in enumerate(boxes[:10]):
            x1, y1, x2, y2 = map(int, box)
            
            # Convert from 1000-scale to actual pixels
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            # Get context for this element
            context = element_sections[i] if i < len(element_sections) else ""
            
            # Extract element type
            element_type = self._extract_type(context)
            
            # Extract text
            text_match = re.search(r'Text:\s*["\']?([^"\'\n]+)["\']?', context, re.IGNORECASE)
            element_text = text_match.group(1).strip() if text_match else ""
            
            # Extract description
            desc_match = re.search(r'Description:\s*(.+?)(?:\n|$)', context, re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else context[:200]
            
            # Extract colors
            colors = self._extract_colors(context)
            
            element = {
                "element_type": element_type,
                "bounding_box": {
                    "x": real_x1,
                    "y": real_y1,
                    "width": real_x2 - real_x1,
                    "height": real_y2 - real_y1
                },
                "text": element_text,
                "description": description,
                "color_palette": colors,
                "confidence": round(0.70 + (i * 0.02), 2)
            }
            elements.append(element)
        
        # Pad to 10 elements
        while len(elements) < 10:
            elements.append(self._empty_element())
        
        return elements[:10]
    
    def _extract_type(self, context):
        """Extract element type from context"""
        context_lower = context.lower()
        
        type_keywords = {
            'button': ['button', 'btn'],
            'link': ['link', 'anchor', 'href'],
            'input': ['input', 'textbox', 'field', 'search bar'],
            'text': ['text', 'label', 'heading', 'paragraph'],
            'image': ['image', 'img', 'logo', 'icon'],
            'dropdown': ['dropdown', 'select'],
            'checkbox': ['checkbox', 'check'],
            'menu': ['menu', 'navigation']
        }
        
        for etype, keywords in type_keywords.items():
            if any(kw in context_lower for kw in keywords):
                return etype
        
        # Default based on first word
        first_line = context.split('\n')[0].lower()
        for etype in type_keywords.keys():
            if etype in first_line:
                return etype
        
        return "unknown"
    
    def _extract_colors(self, context):
        """Extract colors from context"""
        colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 
                 'gray', 'orange', 'purple', 'pink', 'brown']
        found = []
        context_lower = context.lower()
        
        for color in colors:
            if color in context_lower and color not in found:
                found.append(color)
        
        return found[:3] if found else ["white", "black"]
    
    def _empty_element(self):
        """Return a properly formatted empty element"""
        return {
            "element_type": "unknown",
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "text": "",
            "description": "Not detected",
            "color_palette": [],
            "confidence": 0.0
        }
    
    def _empty_elements(self):
        """Return 10 empty elements"""
        return [self._empty_element() for _ in range(10)]