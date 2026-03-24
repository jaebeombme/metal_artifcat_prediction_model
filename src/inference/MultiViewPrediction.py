import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MultiViewResNet50, MultiViewDatasetLoader, config

def predict(model, dataloader):
    predictions = {}

    with torch.no_grad():
        for ap_images, lat_images, img_ids in tqdm(dataloader, desc=f'Multi view model prediction'):
            ap_images, lat_images = ap_images.to(config.DEVICE), lat_images.to(config.DEVICE)
            outputs = model(ap_images, lat_images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.4).astype(int)

            for img_id, pred, prob in zip(img_ids, preds.flatten(), probs.flatten()):
                predictions[img_id] = {
                    'pred': int(pred),
                    'prob': prob * 100
                }

    return predictions

def run(image_dict):
    pred_loader = MultiViewDatasetLoader(image_dict)

    model = MultiViewResNet50().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MULTIVIEW__MODEL_PATH, map_location=config.DEVICE, weights_only=True))
    model.eval()

    predictions = predict(model, pred_loader)

    return predictions

# if __name__ == '__main__':