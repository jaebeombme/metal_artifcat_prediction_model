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

from .SingleViewPrediction import predict
from src import SingleViewResNet50, SingleViewDatasetLoader, config

def run(image_dict):
    pred_loader = SingleViewDatasetLoader(image_dict)

    ap_model = SingleViewResNet50().to(config.DEVICE)
    ap_model.load_state_dict(torch.load(config.AP_MODEL_PATH, weights_only=True))
    ap_model.eval()

    lat_model = SingleViewResNet50().to(config.DEVICE)
    lat_model.load_state_dict(torch.load(config.LAT_MODEL_PATH, weights_only=True))
    lat_model.eval()

    # 예측 수행
    ap_predictions = predict(ap_model, pred_loader, img_type='AP')
    lat_predictions = predict(lat_model, pred_loader, img_type='LAT')

    # 앙상블 결과 저장
    ensemble_predictions = {}

    for img_id in ap_predictions.keys():
        if img_id not in lat_predictions:
            continue  # LAT 예측이 없는 경우 스킵

        ap_prob = ap_predictions[img_id]['prob'] / 100
        lat_prob = lat_predictions[img_id]['prob'] / 100
        ensemble_prob = (ap_prob + lat_prob) / 2
        ensemble_pred = 1 if ensemble_prob > 0.5 else 0

        ensemble_predictions[img_id] = {
            'pred': int(ensemble_pred),
            'prob': ensemble_prob * 100
        }

    return ensemble_predictions
