import os
from PIL import Image
from src import EnsemblePrediction, MaskingCrop, MultiViewPrediction, SingleViewPrediction
from src.eval.metrics import *
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__name__).resolve().parent

raw_image_dir=None
raw_image_dir=f"{BASE_DIR}/data/external/EP_x-ray_sorted_new"

MODE = "external_new" # dev, test, external

processed_image_dir=f"{BASE_DIR}/data/all/data/{MODE}"

result_path=f"{BASE_DIR}/data/all/result/{MODE}"

label_json=f"{BASE_DIR}/data/all/json/{MODE}_labels.json"   # json file (including true_label)

save_path=f"{result_path}/{MODE}_summary.xlsx"


image_dict = None

if not raw_image_dir is None:
    # Image Processing
    image_dict = MaskingCrop.run(image_dir=raw_image_dir, save_dir=processed_image_dir)
else:
    image_dict = defaultdict(list)
    image_dir = processed_image_dir
    for patient_folder in sorted(os.listdir(image_dir)):
        patient_path = os.path.join(image_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for image_file in sorted(os.listdir(patient_path)):
            image_path = os.path.join(patient_path, image_file)

            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    image = Image.open(image_path)
                    image_dict[patient_folder].append(image)
                except IOError:
                    print(f"Can't open image file: {image_path}")
            
ap_predictions, lat_predicitions = SingleViewPrediction.run(image_dict=image_dict)
multi_view_predictions = MultiViewPrediction.run(image_dict=image_dict)
ensemble_predictions = EnsemblePrediction.run(image_dict=image_dict)

print("======AP MODEL======")
plot_confusion_matrix(ap_predictions, label_json, result_path, "AP")
plot_classification_report(ap_predictions, label_json, result_path, "AP")
plot_roc_auc(ap_predictions, label_json, result_path, "AP")

print("======LAT MODEL======")
plot_confusion_matrix(lat_predicitions, label_json, result_path, "LAT")
plot_classification_report(lat_predicitions, label_json, result_path, "LAT")
plot_roc_auc(lat_predicitions, label_json, result_path, "LAT")

print("======Multi View MODEL======")
plot_confusion_matrix(multi_view_predictions, label_json, result_path, "Multiview")
plot_classification_report(multi_view_predictions, label_json, result_path, "Multiview")
plot_roc_auc(multi_view_predictions, label_json, result_path, "Multiview")

print("======Ensemble MODEL======")
plot_confusion_matrix(ensemble_predictions, label_json, result_path, "Ensemble")
plot_classification_report(ensemble_predictions, label_json, result_path, "Ensemble")
plot_roc_auc(ensemble_predictions, label_json, result_path, "Ensemble")

import pandas as pd
import json

def save_predictions_to_excel(
    ap_predictions,
    lat_predictions,
    multi_predictions,
    ensemble_predictions,
    label_json,
    save_path
):
    # 1. GT 불러오기
    if isinstance(label_json, str):
        with open(label_json, 'r') as f:
            label_data = json.load(f)
    else:
        label_data = label_json

    id_to_gt = {item['ID']: int(item['label']) for item in label_data}

    # 2. 모든 ID 수집
    all_ids = set(id_to_gt.keys()) & set(ap_predictions.keys()) & set(lat_predictions.keys()) & \
              set(multi_predictions.keys()) & set(ensemble_predictions.keys())

    # 3. 테이블 구성
    rows = []
    for pid in sorted(all_ids):
        row = {
            "ID": pid,
            "GT": id_to_gt[pid],
            "AP_Pred": ap_predictions[pid]['pred'],
            "AP_Prob": ap_predictions[pid]['prob'],
            "LAT_Pred": lat_predictions[pid]['pred'],
            "LAT_Prob": lat_predictions[pid]['prob'],
            "MultiView_Pred": multi_predictions[pid]['pred'],
            "MultiView_Prob": multi_predictions[pid]['prob'],
            "Ensemble_Pred": ensemble_predictions[pid]['pred'],
            "Ensemble_Prob": ensemble_predictions[pid]['prob']
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 4. 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_excel(save_path, index=False)
    print(f"[INFO] Prediction results saved to: {save_path}")

save_predictions_to_excel(
    ap_predictions=ap_predictions,
    lat_predictions=lat_predicitions,
    multi_predictions=multi_view_predictions,
    ensemble_predictions=ensemble_predictions,
    label_json=label_json,
    save_path=save_path
)