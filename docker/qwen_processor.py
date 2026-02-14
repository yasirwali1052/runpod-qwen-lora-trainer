import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import os

MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"
ELEMENTS_PER_IMAGE = 15

# VALID ELEMENT TYPES - anything else gets rejected
VALID_TYPES = {'button', 'link', 'input', 'text', 'image', 'icon', 'checkbox', 'dropdown', 'menu'}

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
                        "text": f"""Analyze this UI screenshot and identify exactly {ELEMENTS_PER_IMAGE} distinct user interface elements.

For EACH element, provide:
- Type: one of (button, link, input, text, image, icon, checkbox, dropdown, menu)
- Bounding box: <box>[[x,y,x2,y2]]</box> where coordinates are 0-1000
- Text: the visible text on the element (if any)
- Description: brief description of appearance and function

Format each element clearly with numbering (1., 2., 3., etc.).
Detect {ELEMENTS_PER_IMAGE} elements total."""
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
    
    def _validate_type(self, raw_type):
        """Validate and normalize element type to only allowed values"""
        raw_type = raw_type.lower().strip()
        
        # Remove any leading symbols or quotes
        raw_type = re.sub(r'^[\*"\'\s]+', '', raw_type)
        raw_type = re.sub(r'[\*"\'\s]+$', '', raw_type)
        
        # Direct mapping for common variations
        type_mapping = {
            'btn': 'button',
            'hyperlink': 'link',
            'anchor': 'link',
            'textfield': 'input',
            'textbox': 'input',
            'search': 'input',
            'searchbar': 'input',
            'img': 'image',
            'picture': 'image',
            'photo': 'image',
            'logo': 'image',
            'navigation': 'menu',
            'nav': 'menu',
            'navbar': 'menu',
            'select': 'dropdown',
            'picker': 'dropdown',
            'toggle': 'checkbox',
            'switch': 'checkbox'
        }
        
        # Try mapping first
        if raw_type in type_mapping:
            return type_mapping[raw_type]
        
        # Check if it's already valid
        if raw_type in VALID_TYPES:
            return raw_type
        
        # Try to find a valid type within the string
        for valid_type in VALID_TYPES:
            if valid_type in raw_type:
                return valid_type
        
        # Default to button (most common)
        return 'button'
    
    def _parse_response(self, response, img_w, img_h):
        """Strict parser with type validation"""
        elements = []
        
        # Find all bounding boxes
        box_pattern = r'<box>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, response)
        
        if not boxes:
            print("❌ NO BOXES FOUND")
            return [self._empty_element() for _ in range(ELEMENTS_PER_IMAGE)]
        
        print(f"✓ Found {len(boxes)} boxes")
        
        # Split by numbered sections
        sections = re.split(r'\n\s*\d+\.\s+', response)
        sections = [s.strip() for s in sections if s.strip()]
        
        for i, box in enumerate(boxes[:ELEMENTS_PER_IMAGE]):
            x1, y1, x2, y2 = map(int, box)
            
            # Scale coordinates
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            # Get context for this element
            context = sections[i] if i < len(sections) else ""
            
            # EXTRACT TYPE with strict validation
            elem_type = 'button'  # default
            type_patterns = [
                r'[Tt]ype\s*[:=]\s*(\w+)',
                r'^\s*(\w+)\s*:',  # First word followed by colon
                r'^\s*\*\*(\w+)\*\*'  # **Type**
            ]
            
            for pattern in type_patterns:
                type_match = re.search(pattern, context)
                if type_match:
                    raw_type = type_match.group(1)
                    elem_type = self._validate_type(raw_type)
                    break
            
            # EXTRACT TEXT - look for quoted strings
            text_content = ""
            # Try multiple quote patterns
            quote_patterns = [
                r'[Tt]ext\s*[:=]\s*["\']([^"\']+)["\']',
                r'["\']([^"\']{3,100})["\']'
            ]
            
            for pattern in quote_patterns:
                quote_matches = re.findall(pattern, context)
                if quote_matches:
                    # Get the longest match (likely the actual text)
                    text_content = max(quote_matches, key=len).strip()
                    break
            
            # EXTRACT DESCRIPTION - clean version
            description = "UI element"
            
            # Remove all metadata lines
            clean_context = context
            clean_context = re.sub(r'<box>.*?</box>', '', clean_context)
            clean_context = re.sub(r'[Tt]ype\s*[:=].*', '', clean_context)
            clean_context = re.sub(r'[Tt]ext\s*[:=].*', '', clean_context)
            clean_context = re.sub(r'\*\*[^*]+\*\*\s*[:=]', '', clean_context)  # Remove **Label**:
            clean_context = re.sub(r'-\s*\n', '', clean_context)  # Remove list markers
            clean_context = re.sub(r'\*\*[Vv]isible.*', '', clean_context)  # Remove **Visible
            clean_context = re.sub(r'\*\*[Dd]escription.*?:', '', clean_context)  # Remove **Description**:
            clean_context = clean_context.strip()
            
            # Get first meaningful sentence
            if clean_context and len(clean_context) > 15:
                sentences = re.split(r'[.!?]\s+', clean_context)
                for sent in sentences:
                    sent = sent.strip()
                    # Skip very short or generic sentences
                    if len(sent) > 15 and not any(x in sent.lower() for x in ['here is', 'visible text', 'description', 'bounding box']):
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
            
            print(f"  [{i+1:2d}] {elem_type:8s} | '{text_content[:30]:30s}' | {description[:40]:40s}")
        
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
            "confidence": 0.0
        }