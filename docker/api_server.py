import torch
import gc
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import threading
from qwen_processor import QwenProcessor

app = FastAPI(title="Qwen Vision API", version="1.0.0")
processor = None
gpu_lock = threading.Lock()

INPUT_FOLDER = "/workspace/input"
OUTPUT_FOLDER = "/workspace/output"

@app.get("/")
async def root():
    return {"status": "running", "message": "Qwen Vision API"}

@app.on_event("startup")
async def startup():
    global processor
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("Loading Qwen model...")
    try:
        processor = QwenProcessor()
        print("Model loaded successfully")
    except Exception as e:
        print(f"STARTUP ERROR: {traceback.format_exc()}")
        raise

@app.post("/process")
async def process_single_image(image: UploadFile = File(...)):
    try:
        file_path = os.path.join(INPUT_FOLDER, image.filename)
        
        with open(file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        print(f"Processing: {image.filename}")
        
        with gpu_lock:
            result = processor.process_image(file_path)
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Completed: {image.filename}")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"PROCESSING ERROR:\n{error_trace}")
        return JSONResponse(
            content={
                "error": str(e),
                "traceback": error_trace,
                "file": image.filename if image else "unknown"
            },
            status_code=500
        )

@app.get("/health")
async def health_check():
    try:
        vram = f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "No GPU"
        return {
            "status": "healthy",
            "model": "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
            "vram": vram
        }
    except:
        return {"status": "healthy", "model": "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")