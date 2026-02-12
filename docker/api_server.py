import torch
import gc
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import glob
import threading
from qwen_processor import QwenProcessor

app = FastAPI(title="Qwen Vision API", version="1.0.0")
processor = None
# Lock to prevent multiple requests from hitting the GPU simultaneously
gpu_lock = threading.Lock()

# Standard RunPod/Docker paths
INPUT_FOLDER = "/workspace/input"
OUTPUT_FOLDER = "/workspace/output"

@app.on_event("startup")
async def startup():
    global processor
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("Loading model into VRAM...")
    processor = QwenProcessor()
    print("Model loaded and ready.")

@app.post("/process")
async def process_single_image(image: UploadFile = File(...)):
    try:
        # Using INPUT_FOLDER instead of /tmp for disk-based storage safety
        file_path = os.path.join(INPUT_FOLDER, image.filename)
        
        with open(file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Use a lock to ensure only one image hits the GPU at a time
        with gpu_lock:
            result = processor.process_image(file_path)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Optional: Force memory cleanup after processing
        torch.cuda.empty_cache()
        gc.collect()
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/process-folder")
async def process_folder(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_all_images)
    return {"status": "processing started"}

def process_all_images():
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not image_files:
        return
    
    for img_path in image_files:
        try:
            with gpu_lock:
                result = processor.process_image(img_path)
            
            filename = os.path.basename(img_path)
            output_file = os.path.join(
                OUTPUT_FOLDER, 
                f"{os.path.splitext(filename)[0]}.json"
            )
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Cleanup VRAM between images in a batch
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "vram_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    }

@app.get("/results")
async def get_results():
    json_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.json"))
    results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                results.append(json.load(f))
        except:
            continue
    return {"count": len(results), "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)