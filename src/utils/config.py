import torch
from pathlib import Path

# Image processing setting
REAL_WORLD_SIZE = 240

# Current Directory Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Input & Ouput Directory 
INPUT_DIR = BASE_DIR / "data" / "raw"
GENERATED_DIR = BASE_DIR / "data" / "generated" # for saving original image and masking image
PROCESSED_DIR = BASE_DIR / "data" / "processed" # saving path for cropped image
OUTPUT_BASE = BASE_DIR / "outputs"

# model setting
MASKING_MODEL_PATH = BASE_DIR / "checkpoints" / "Masking_Model.pth"
MULTIVIEW__MODEL_PATH = BASE_DIR / "checkpoints" / "MultiViewModel.pth"
AP_MODEL_PATH  = BASE_DIR / "checkpoints" / "AP.pth"
LAT_MODEL_PATH  = BASE_DIR / "checkpoints" / "LAT.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'