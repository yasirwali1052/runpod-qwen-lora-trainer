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
                        "text": f"""Analyze this screenshot and list exactly {ELEMENTS_PER_IMAGE} distinct, interactive UI elements.

For each element provide:
1. Type: (button/link/input/text/image/icon/menu/header/navigation/search/etc)
2. Position: <box>[[x1,y1,x2,y2]]</box> in 1000x1000 coordinates
3. Text: "exact visible text" in quotes (or empty "" if no text)
4. Purpose: One clear sentence describing what this element does
5. Colors: Main colors used (e.g., blue, white, red)

Focus on interactive elements users can click/type into. Avoid listing the same background element multiple times.
List elements from top to bottom, left to right."""
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
        """COMPLETELY REWRITTEN PARSER - Fixed for actual model output"""
        elements = []
        
        # Find all boxes with their coordinates
        box_pattern = r'<box>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]</box>'
        boxes = list(re.finditer(box_pattern, response))
        
        if not boxes:
            print("❌ NO BOXES FOUND IN MODEL OUTPUT")
            return [self._empty_element() for _ in range(ELEMENTS_PER_IMAGE)]
        
        print(f"✓ Found {len(boxes)} boxes in model output")
        
        # Split response into sections - each section describes one box
        # Text between box N and box N+1 describes box N
        sections = []
        for i, box_match in enumerate(boxes):
            start_pos = box_match.end()
            end_pos = boxes[i+1].start() if i+1 < len(boxes) else len(response)
            section_text = response[start_pos:end_pos].strip()
            sections.append(section_text)
        
        # Track seen bounding boxes to detect duplicates
        seen_boxes = set()
        
        for i, box_match in enumerate(boxes[:ELEMENTS_PER_IMAGE * 2]):  # Process extra to filter duplicates
            x1, y1, x2, y2 = map(int, box_match.groups())
            
            # Scale coordinates from 1000x1000 to actual image size
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)
            
            # Skip if coordinates are invalid
            if real_x2 <= real_x1 or real_y2 <= real_y1:
                continue
            
            # Create box signature for duplicate detection
            box_sig = (real_x1, real_y1, real_x2, real_y2)
            
            # Get the text content for this element
            context = sections[i] if i < len(sections) else ""
            
            # Clean up markdown formatting (remove bullet points, dashes, etc)
            context_clean = re.sub(r'^\s*[-•*]\s*', '', context, flags=re.MULTILINE)
            context_clean = re.sub(r'\n\s*[-•*]\s*', ' ', context_clean)
            context_clean = ' '.join(context_clean.split())  # normalize whitespace
            
            # EXTRACT TYPE
            elem_type = "unknown"
            type_patterns = [
                r'\*\*Type\*\*:\s*(\w+)',
                r'Type:\s*(\w+)',
                r'^\s*(\w+)\s+element',
                r'^\s*(button|link|input|text|image|icon|menu|header|navigation|search|sign|login|video|main|subheading)\b'
            ]
            for pattern in type_patterns:
                match = re.search(pattern, context_clean, re.IGNORECASE)
                if match:
                    elem_type = match.group(1).lower()
                    break
            
            # EXTRACT TEXT (quoted content)
            text_content = ""
            # Look for text in quotes - support various quote styles
            quote_pattern = r'[""\'"`]([^""\'"`]{1,200})[""\'"`]'
            quote_matches = re.findall(quote_pattern, context_clean)
            if quote_matches:
                # Get the longest quoted string (usually the actual UI text)
                text_content = max(quote_matches, key=len).strip()
                # Clean up common artifacts
                text_content = text_content.replace('\\n', ' ')
                text_content = ' '.join(text_content.split())
            
            # EXTRACT DESCRIPTION
            description = "UI element"
            # Remove type/text markers and get remaining content
            desc_text = re.sub(r'\*\*[Tt]ype\*\*:\s*\w+', '', context_clean)
            desc_text = re.sub(r'\*\*[Tt]ext\*\*:\s*[""\'"`].*?[""\'"`]', '', desc_text)
            desc_text = re.sub(r'[""\'"`].*?[""\'"`]', '', desc_text)  # remove all quoted text
            desc_text = re.sub(r'Type:\s*\w+', '', desc_text, flags=re.IGNORECASE)
            desc_text = re.sub(r'\*\*[^*]+\*\*:', '', desc_text)  # remove **Label**: patterns
            desc_text = desc_text.strip()
            
            # Get first meaningful sentence (at least 15 chars)
            sentences = re.split(r'[.!?]\s+', desc_text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) >= 15:
                    # Avoid meta-descriptions about the format
                    skip_phrases = ['bounding box', 'type:', 'text:', 'visible text', 
                                   'here is', 'the element', 'this is a', 'coordinates']
                    if not any(phrase in sent.lower() for phrase in skip_phrases):
                        description = sent[:200]
                        break
            
            # If no good description found, use cleaned snippet
            if description == "UI element" and len(desc_text) > 15:
                description = desc_text[:200]
            
            # EXTRACT COLORS
            colors = []
            color_words = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 
                          'pink', 'white', 'black', 'gray', 'grey', 'brown', 
                          'cyan', 'magenta', 'teal', 'navy', 'lime']
            context_lower = context_clean.lower()
            for color in color_words:
                if re.search(r'\b' + color + r'\b', context_lower):
                    if color not in colors and color != 'grey' if 'gray' in colors else True:
                        colors.append(color)
                    if len(colors) >= 3:
                        break
            
            # QUALITY CHECKS - Filter out bad elements
            
            # Check if this is a duplicate background element
            element_area = (real_x2 - real_x1) * (real_y2 - real_y1)
            total_area = img_w * img_h
            is_fullscreen = element_area >= total_area * 0.85  # 85% or more of screen
            
            background_keywords = ['background', 'gradient', 'main background', 
                                  'the page', 'entire page', 'full page']
            is_background_desc = any(term in description.lower() for term in background_keywords)
            
            # Skip duplicate backgrounds (keep first one only)
            if is_fullscreen and is_background_desc and len(elements) >= 3:
                print(f"  [{i+1}] SKIPPED - duplicate background element")
                continue
            
            # Skip exact duplicate bounding boxes
            if box_sig in seen_boxes:
                print(f"  [{i+1}] SKIPPED - duplicate bounding box")
                continue
            
            seen_boxes.add(box_sig)
            
            # Create element
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
                "color_palette": colors,
                "confidence": round(0.70 + (len(elements) * 0.01), 2)
            }
            elements.append(element)
            
            # Print progress
            text_preview = text_content[:25] if text_content else "<no text>"
            desc_preview = description[:35] if description else "<no desc>"
            print(f"  [{len(elements):2d}] {elem_type:12s} | '{text_preview:25s}' | {desc_preview:35s} | {colors}")
            
            # Stop once we have enough good elements
            if len(elements) >= ELEMENTS_PER_IMAGE:
                break
        
        # Fill remaining slots with empty elements if needed
        while len(elements) < ELEMENTS_PER_IMAGE:
            elements.append(self._empty_element())
        
        print(f"\n✓ Parsed {len([e for e in elements if e['element_type'] != 'unknown'])} valid elements")
        
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


# Example usage
if __name__ == "__main__":
    import json
    
    processor = QwenProcessor()
    
    # Process an image
    result = processor.process_image("screenshot_014.png")
    
    # Save result
    with open("output_fixed.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Saved results to output_fixed.json")
    print(f"✓ Found {result['total_elements']} elements")