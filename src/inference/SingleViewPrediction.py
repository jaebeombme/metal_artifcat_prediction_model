import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import SingleViewResNet50, SingleViewDatasetLoader, config

def predict(model, dataloader, img_type):
    predictions = {}

    with torch.no_grad():
        for ap_images, lat_images, img_ids in tqdm(dataloader, desc=f'{img_type} view model prediction'):
            ap_images, lat_images = ap_images.to(config.DEVICE), lat_images.to(config.DEVICE)
            
            if img_type == 'AP':
                outputs = model(ap_images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            elif img_type == 'LAT':
                outputs = model(lat_images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

            for img_id, pred, prob in zip(img_ids, preds.flatten(), probs.flatten()):
                predictions[img_id] = {
                    'pred': int(pred),
                    'prob': prob * 100
                }

    return predictions

def run(image_dict):
    pred_loader = SingleViewDatasetLoader(image_dict)
    ap_model = SingleViewResNet50().to(config.DEVICE)
    ap_model.load_state_dict(torch.load(config.AP_MODEL_PATH, weights_only=True))
    ap_model.eval()

    lat_model = SingleViewResNet50().to(config.DEVICE)
    lat_model.load_state_dict(torch.load(config.LAT_MODEL_PATH, weights_only=True))
    lat_model.eval()

    ap_predictions = predict(ap_model, pred_loader, img_type='AP')
    lat_predictions = predict(lat_model, pred_loader, img_type='LAT')

    return ap_predictions, lat_predictions