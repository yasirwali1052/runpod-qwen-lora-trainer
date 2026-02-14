import os
import shutil

input_dir = "data/input"
output_dir = "data/output"
approved_dir = "data/approved"

os.makedirs(approved_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img_file in image_files:
    base_name = os.path.splitext(img_file)[0]
    json_file = base_name + ".json"
    
    img_src = os.path.join(input_dir, img_file)
    json_src = os.path.join(output_dir, json_file)
    
    if os.path.exists(json_src):
        shutil.copy(img_src, approved_dir)
        shutil.copy(json_src, approved_dir)
        print(f"Copied {base_name}")
    else:
        print(f"Skipped {base_name} - JSON not found")

print(f"\nDone! Check {approved_dir}")