from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_confusion_matrix(predictions, label_json, save_path, model_type):
    os.makedirs(save_path, exist_ok=True)

    if isinstance(label_json, str):
        with open(label_json, 'r') as f:
            label_data = json.load(f)
    else:
        label_data = label_json

    id_to_label = {item['ID']: item['label'] for item in label_data}

    y_true = []
    y_pred = []

    for pid, pred_info in predictions.items():
        if pid in id_to_label:
            y_true.append(id_to_label[pid])
            y_pred.append(pred_info['pred'])

    y_true = [int(label) for label in y_true]
    y_pred = [int(label) for label in y_pred]

    classes = sorted(set(y_true) | set(y_pred))
    class_names = [str(c) for c in classes]

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_path, f"confusion_matrix_{model_type}.png"))
    plt.close()

    print("[Confusion Matrix]")
    print(cm)


def plot_classification_report(predictions, label_json, save_path, model_type):
    os.makedirs(save_path, exist_ok=True)

    if isinstance(label_json, str):
        with open(label_json, 'r') as f:
            label_data = json.load(f)
    else:
        label_data = label_json

    id_to_label = {item['ID']: item['label'] for item in label_data}

    y_true = []
    y_pred = []

    for pid, pred_info in predictions.items():
        if pid in id_to_label:
            y_true.append(id_to_label[pid])
            y_pred.append(pred_info['pred'])

    y_true = [int(label) for label in y_true]
    y_pred = [int(label) for label in y_pred]

    classes = sorted(set(y_true) | set(y_pred))
    class_names = [str(c) for c in classes]

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n[Classification Report]")
    print(report)

    with open(os.path.join(save_path, f"classification_report_{model_type}.txt"), "w") as f:
        f.write(report)

def plot_roc_auc(predictions, label_json, save_path, model_type):
    os.makedirs(save_path, exist_ok=True)

    # Load label data
    if isinstance(label_json, str):
        with open(label_json, 'r') as f:
            label_data = json.load(f)
    else:
        label_data = label_json

    id_to_label = {item['ID']: int(item['label']) for item in label_data}

    y_true = []
    y_prob = []

    for pid, pred_info in predictions.items():
        if pid in id_to_label:
            y_true.append(id_to_label[pid])
            y_prob.append(pred_info['prob'])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    classes = sorted(set(y_true))

    # Binarize labels for multi-class
    y_true_bin = label_binarize(y_true, classes=classes)

    if y_prob.ndim == 1 or y_prob.shape[1] == 1 or len(classes) == 2:
        # Binary classification
        if y_prob.ndim == 2:
            y_prob_bin = y_prob[:, -1]
        else:
            y_prob_bin = y_prob
        score = roc_auc_score(y_true, y_prob_bin)
        fpr, tpr, _ = roc_curve(y_true, y_prob_bin)

        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_type} (AUC = {score:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_type}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, f"roc_auc_curve_{model_type}.png"))
        plt.close()

        print(f"\n[ROC AUC Score] ({model_type} - Binary): {score:.4f}")

    else:
        # Multi-class ROC-AUC
        fpr, tpr, roc_auc = {}, {}, {}
        for i, c in enumerate(classes):
            fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[c] = auc(fpr[c], tpr[c])

        plt.figure()
        for c in classes:
            plt.plot(fpr[c], tpr[c], label=f"Class {c} (AUC = {roc_auc[c]:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC-AUC Curve - {model_type}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, f"roc_auc_curve_{model_type}.png"))
        plt.close()

        print(f"\n[ROC AUC Scores] ({model_type} - Multi-Class):")
        for c in classes:
            print(f"  Class {c}: AUC = {roc_auc[c]:.4f}")