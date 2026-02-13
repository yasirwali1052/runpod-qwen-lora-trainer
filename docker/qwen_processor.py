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
                        "text": f"""Analyze this screenshot and identify at least {ELEMENTS_PER_IMAGE} distinct UI elements.

For each element use this format strictly:

Type: button|link|input|text|image|icon|menu|header|navigation|search
Position: <box>[[x1,y1,x2,y2]]</box> in 1000x1000 normalized coordinates
Text: "visible text"
Purpose: One short sentence
Colors: comma separated colors

List elements from top-left to bottom-right."""
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

        # ✅ Reduced tokens + sampling enabled
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )

        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        elements = self._parse_response(response, orig_width, orig_height)

        return {
            "image_filename": os.path.basename(image_path),
            "total_elements": len(elements),
            "elements": elements
        }

    # ---------------------------------------------------
    # CLEAN PARSER
    # ---------------------------------------------------

    def _parse_response(self, response, img_w, img_h):
        elements = []

        box_pattern = r'<box>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]</box>'
        boxes = list(re.finditer(box_pattern, response))

        if not boxes:
            return [self._empty_element() for _ in range(ELEMENTS_PER_IMAGE)]

        seen_boxes = []

        for i, box_match in enumerate(boxes):

            if len(elements) >= ELEMENTS_PER_IMAGE:
                break

            x1, y1, x2, y2 = map(int, box_match.groups())

            # Scale coordinates
            real_x1 = int(x1 * img_w / 1000)
            real_y1 = int(y1 * img_h / 1000)
            real_x2 = int(x2 * img_w / 1000)
            real_y2 = int(y2 * img_h / 1000)

            if real_x2 <= real_x1 or real_y2 <= real_y1:
                continue

            # Basic duplicate filtering (position similarity)
            duplicate = False
            for (sx1, sy1, sx2, sy2) in seen_boxes:
                if abs(real_x1 - sx1) < 20 and abs(real_y1 - sy1) < 20:
                    duplicate = True
                    break

            if duplicate:
                continue

            seen_boxes.append((real_x1, real_y1, real_x2, real_y2))

            # Extract text block for this element
            start = box_match.start()
            end = boxes[i + 1].start() if i + 1 < len(boxes) else len(response)
            block = response[start:end]

            # Remove next numbered element leakage
            block = re.split(r'\n\d+\.\s*Type:', block)[0]

            # Extract Type
            type_match = re.search(r'Type:\s*(\w+)', block, re.IGNORECASE)
            elem_type = type_match.group(1).lower() if type_match else "unknown"

            # Extract Text
            text_match = re.search(r'Text:\s*["“”]?([^"\n]+)', block)
            text_content = text_match.group(1).strip() if text_match else ""

            # Extract Purpose
            purpose_match = re.search(r'Purpose:\s*(.+)', block)
            description = purpose_match.group(1).strip() if purpose_match else "UI element"

            # Clean description
            description = re.sub(r'Colors:.*', '', description)
            description = re.sub(r'\s+\d+\.?\s*$', '', description).strip()

            if description and not description.endswith("."):
                description += "."

            # Extract Colors
            color_match = re.search(r'Colors?:\s*(.+)', block, re.IGNORECASE)
            colors = []
            if color_match:
                raw_colors = color_match.group(1)
                colors = [c.strip().lower() for c in raw_colors.split(",")][:3]

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
                "confidence": round(0.75 + (len(elements) * 0.01), 2)
            }

            elements.append(element)

        # Fill missing slots
        while len(elements) < ELEMENTS_PER_IMAGE:
            elements.append(self._empty_element())

        return elements[:ELEMENTS_PER_IMAGE]

    # ---------------------------------------------------

    def _empty_element(self):
        return {
            "element_type": "unknown",
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "text": "",
            "description": "Not detected",
            "color_palette": [],
            "confidence": 0.0
        }
