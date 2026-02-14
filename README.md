# Qwen-VL RunPod Pipeline - Complete Setup Guide


## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Initial Setup](#initial-setup)
4. [Getting RunPod API Key](#getting-runpod-api-key)
5. [Running the Pipeline](#running-the-pipeline)
6. [LoRA Fine-Tuning](#lora-fine-tuning)
7. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project helps you:

1. **Process screenshots** on powerful RunPod GPUs using Qwen2-VL AI model
2. **Extract UI elements** with bounding boxes and descriptions automatically
3. **Review and approve** results manually on your computer
4. **Fine-tune the model** with approved data to improve accuracy over time

**No coding knowledge required** - just follow the steps below.

---

##  Project Structure

```
runpod-qwen-lora-trainer/
│
├── data/
│   ├── input/          ← Put your screenshots here
│   ├── output/         ← Processed results appear here
│   └── approved/       ← Copy reviewed good results here
│
├── docker/             ← Docker container files (don't touch)
│
├── weights/            ← Trained model improvements saved here
│
├── main.py             ← Main control program
├── lora_trainer.py     ← Training program
└── copy_to_approved.py ← Helper to copy files
```

---

##  Initial Setup

### Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/yasirwali1052/runpod-qwen-lora-trainer.git
cd runpod-qwen-lora-trainer
```

Or download the ZIP file from GitHub and extract it.

---

### Step 2: Create Python Environment

A virtual environment keeps this project's dependencies separate from your system.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal. This means the environment is active.

---

### Step 3: Install Dependencies

**For the main program (in root folder):**

```bash
pip install -r requirements.txt
```

**For LoRA training (optional, only if you want to fine-tune):**

```bash
pip install -r lora_requirements_cpu.txt
```

This will take 5-10 minutes to download and install everything.

---

### Step 4: Create Required Folders

Create the approved data folder:

```bash
mkdir data\approved
```

Or create it manually in File Explorer.

---

##  Getting RunPod API Key

You need a RunPod account and API key to use cloud GPUs.

### Step 1: Create RunPod Account

1. Go to [https://runpod.io](https://runpod.io)
2. Click "Sign Up" and create an account
3. Add credit to your account (minimum $10 recommended)

### Step 2: Generate API Key

1. Log in to RunPod
2. Click on your profile (top right)
3. Select "Settings"
4. Go to "API Keys" tab
5. Click "Create API Key"
6. Give it a name (e.g., "Qwen Pipeline")
7. Copy the key (looks like: `ABC123XYZ456...`)

**Important:** Save this key somewhere safe. You'll need it in the next step.

---

##  Running the Pipeline

### Step 1: Start the Manager

In your terminal (with venv activated), run:

```bash
python main.py
```

### Step 2: Enter API Key (First Time Only)

When you run it the first time, you'll see:

```
Enter RunPod API Key: 
```

Paste your RunPod API key and press Enter. It will be saved for future use.

---

### Step 3: Main Menu

You'll see this menu:

```
==================================================
QWEN RUNPOD MANAGER
==================================================
1. Create Pod
2. View Pods & Endpoints
3. Process Images
4. Stop Pod
5. Terminate Pod
6. Exit

Select: 
```

---

## Working with RunPod

### Creating a GPU Pod

**Step 1:** Type `1` and press Enter

**Step 2:** Choose your GPU

You'll see a list of available GPUs:

```
Available GPUs:
1. MI300X (192GB)
2. A100 PCIe (80GB)
3. A100 SXM (80GB)
4. A30 (24GB)
5. A40 (48GB)
...
16. RTX 4090 (24GB)
...

Select GPU number: 
```

**Recommended for beginners:** 
- RTX 4090 (option 16) - Good balance of speed and cost
- A40 (option 5) - Reliable and affordable

Type the number (e.g., `5` for A40) and press Enter.

**Step 3:** Name Your Pod

```
Pod name [Qwen-Worker]: 
```

You can press Enter to use default name, or type your own (e.g., `Screenshot-Processor`).

**Step 4:** Set Disk Size

```
Disk GB [50]: 
```

Press Enter for default 50GB (recommended for most uses).

**Step 5:** Wait for Pod Creation

You'll see:

```
Pod created: 2r76w74z1layn8
Wait 5-10 minutes for model to download and load into VRAM
```

**Important:** Write down this Pod ID. You'll need it later.

The first time takes 10-15 minutes because it downloads the 7GB AI model.

---

### Checking Pod Status

**Step 1:** From main menu, type `2` and press Enter

You'll see your pods:

```
ID                 NAME                 STATUS       GPU
--------------------------------------------------------------------------------
2r76w74z1layn8     Qwen-Worker         RUNNING      RTX 4090
  API: https://2r76w74z1layn8-8000.proxy.runpod.net

```

**Status meanings:**
- `RUNNING` - Ready to use
- `INIT` - Still starting up, wait a few minutes
- `STOPPED` - Paused, not using credits

**API URL:** This is how your computer talks to the GPU. It appears automatically when ready.

---

### Processing Screenshots

**Step 1:** Add Your Screenshots

Put your screenshot files in the `data/input/` folder.

Supported formats: `.png`, `.jpg`, `.jpeg`

Example:
```
data/input/
├── screenshot_001.png
├── screenshot_002.png
└── screenshot_003.png
```

**Step 2:** Start Processing

From main menu, type `3` and press Enter

**Step 3:** Enter Pod ID

```
Enter Pod ID: 
```

Type your pod ID (e.g., `2r76w74z1layn8`) and press Enter.

**Step 4:** Enter API Endpoint (or Skip)

```
Enter endpoint URL (or press Enter to auto-detect): 
```

Just press Enter. It will find the URL automatically.

**Step 5:** Watch Progress

You'll see:

```
Processing 3 images via https://2r76w74z1layn8-8000.proxy.runpod.net
[1/3] screenshot_001.png
   Saved: data/output/screenshot_001.json
[2/3] screenshot_002.png
   Saved: data/output/screenshot_002.json
[3/3] screenshot_003.png
   Saved: data/output/screenshot_003.json

Results saved in data/output/
```

Each image takes 30-60 seconds to process.

---

### Viewing Results

Open `data/output/` folder. You'll find JSON files with extracted information:

```json
{
  "image_filename": "screenshot_001.png",
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "total_elements": 15,
  "elements": [
    {
      "element_type": "button",
      "bounding_box": { "x": 120, "y": 45,"width": 80, "height": 35},
      "text": "Sign In",
      "description": "Login button in top right corner",
      "confidence": 0.70
    }
  ]
}
```

---

### Stopping and Terminating Pods

**Stop Pod (Pause):**
- Type `4` from main menu
- Enter Pod ID
- Pod pauses, you stop paying, but data stays

**Terminate Pod (Delete):**
- Type `5` from main menu
- Enter Pod ID
- Pod deleted completely, no more charges

**Important:** Always terminate pods when done to avoid charges!

---

## 🎓 LoRA Fine-Tuning

Fine-tuning makes the AI better at YOUR specific screenshots over time.

### Step 1: Review Your Results

1. Open `data/output/` folder
2. Look at each JSON file
3. Check if the extracted information is correct

**Good result example:** All UI elements found, accurate bounding boxes, correct labels

**Bad result example:** Missing elements, wrong positions, incorrect element types

---

### Step 2: Copy Approved Data

Only copy the GOOD results to the approved folder.

**Manual Method:**

For each good result:
1. Copy the image from `data/input/` to `data/approved/`
2. Copy matching JSON from `data/output/` to `data/approved/`

**Example:**
```bash
copy data\input\screenshot_001.png data\approved\
copy data\output\screenshot_001.json data\approved\
```

**Automatic Method (Copy All):**

If most results are good, use the helper script:

```bash
python copy_to_approved.py
```

This copies all pairs automatically. Then manually delete the bad ones from `data/approved/`.

---

### Step 3: Start Training

Once you have at least 5-10 approved pairs in `data/approved/`, run:

```bash
python lora_trainer.py
```

You'll see:

```
============================================================
QWEN2-VL LoRA FINE-TUNING (CPU MODE)
============================================================

Initializing trainer...
Loading base model (this may take 5-10 minutes)...
```

**First time:** Downloads huge model, takes 10-30 minutes

**Step 4:** Enter Training Parameters

```
Number of epochs [1]: 
```

Press Enter for 1 epoch (recommended for first training).

```
Learning rate [2e-4]: 
```

Press Enter for default.

```
Output name [lora_v1]: 
```

Type a name like `lora_v1` or press Enter for default.

**Step 5:** Wait for Training

```
Loaded 8 approved image-label pairs

Training on 8 examples for 1 epoch(s)
WARNING: Training on CPU will be VERY SLOW
Estimated time: 1-3 hours per epoch depending on data size

Starting training...
```

**Important:** Training on CPU is SLOW. Expect 1-3 hours per epoch.

Go get coffee, watch a movie, come back later.

**Step 6:** Training Complete

```
Training complete!
LoRA weights saved to: weights/lora_v1
```

Your improved model is now saved in the `weights/` folder.

---

### Step 4: Continue Training Later

As you get more approved data, you can continue training:

1. Add new approved pairs to `data/approved/`
2. Run `python lora_trainer.py`
3. Choose option 2 to continue from previous training
4. Select which version to continue from

This way the model keeps getting better over time.

---

## Troubleshooting

### "ModuleNotFoundError"

**Problem:** Missing Python packages

**Solution:**
```bash
pip install -r requirements.txt
pip install -r lora_requirements.txt
```

---

### "Pod API endpoint not ready"

**Problem:** Pod still starting up

**Solution:** Wait 5-10 minutes, then try again. First startup is always slow.

---

### "No images in data/input/"

**Problem:** Input folder is empty

**Solution:** Put your screenshot files in `data/input/` folder before processing.

---

### Training is too slow

**Problem:** CPU training takes hours

**Solutions:**
- Use fewer approved pairs (5-10 instead of 50)
- Use only 1 epoch
- Or use Google Colab free GPU (advanced)

**Happy Processing!**