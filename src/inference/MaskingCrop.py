import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import re
import torch
from PIL import Image
from pathlib import Path
import pydicom
from collections import defaultdict
from pydicom.pixel_data_handlers.util import apply_voi_lut

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MaskingModel, config

type_pattern = re.compile(r"(AP|LAT)", re.IGNORECASE)

class DICOMProcessor:
    def __init__(self, dicom_dir, img_height=256, img_width=256, savepath=config.GENERATED_DIR):
        self.dicom_dir = dicom_dir
        self.img_height = img_height
        self.img_width = img_width
        self.savepath = savepath

    def load_dicom_images(self):
        images = []
        filenames = []
        original_images = []
        pixel_spacing_array = []

        for patient_folder in tqdm(sorted(os.listdir(self.dicom_dir)), desc="Loading DICOMs"):
            patient_path = os.path.join(self.dicom_dir, patient_folder)
            if not os.path.isdir(patient_path):
                continue 
            
            for view in ['AP', 'LAT']:
                view_path = os.path.join(patient_path, view)
                if not os.path.isdir(view_path):
                    continue
                
                for dcm_file in sorted(os.listdir(view_path)):
                    if not dcm_file.endswith(".dcm"):
                        continue

                    dcm_path = os.path.join(view_path, dcm_file)

                    new_filename = f"{view}_{patient_folder}.png"
                    dicom_data = pydicom.dcmread(dcm_path)
                    img = dicom_data.pixel_array
                    pixel_spacing = float(dicom_data.get("ImagerPixelSpacing", dicom_data.get("PixelSpacing", [1, 1]))[0])

                    img_voilut = apply_voi_lut(img, dicom_data)
                    img_scaled = ((img_voilut - np.min(img_voilut)) / (np.max(img_voilut) - np.min(img_voilut))) * 65535
                    img_uint16_recovered = img_scaled.astype(np.uint16)

                    path = os.path.join(self.savepath, 'original', patient_folder)
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(path, new_filename), img_uint16_recovered)

                    img_resized = cv2.resize(img_uint16_recovered, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
                    img_array = img_resized.astype(np.float32) / 65535.0

                    original_images.append(img_uint16_recovered)
                    images.append(img_array)
                    filenames.append(new_filename)
                    pixel_spacing_array.append(pixel_spacing)

        return np.array(images), original_images, filenames, pixel_spacing_array

class ImageProcessor:
    def __init__(self, savepath=config.GENERATED_DIR, contrast_scale=1.5):
        self.contrast_scale = contrast_scale
        self.savepath = savepath

    def preprocess_images_to_tensor(self, images):
        image_tensor = torch.from_numpy(images).float().unsqueeze(1)
        if image_tensor.dim() == 3:
            # (C, H, W) → (1, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 2:
            # (H, W) → (1, 1, H, W)
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        mean = image_tensor.mean(dim=(0, 2, 3), keepdim=True)
        std = image_tensor.std(dim=(0, 2, 3), keepdim=True)
        image_tensor = ((image_tensor - mean) / (std + 1e-7)) * self.contrast_scale
        return image_tensor

    def resize_masks(self, original_images, mask_list, filename):
        resized_masks = []
        for original_img, mask, name in tqdm(zip(original_images, mask_list, filename), total=len(filename), desc="Resizing Masks"):
            original_size = (original_img.shape[1], original_img.shape[0])
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            resized_mask = mask_img.resize(original_size, Image.NEAREST).convert("L")
            resized_masks.append(np.array(resized_mask))

            parts = name.replace(".png", "").split("_")
            patient_id, study_date = parts[1], parts[2]
            folder_name = patient_id + '_' + study_date

            path = os.path.join(self.savepath, "resize_mask", folder_name, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            resized_mask.save(path, format="png")

        return resized_masks

class MaskProcessor:
    def __init__(self, savepath, real_world_size=config.REAL_WORLD_SIZE):
        self.real_world_size = real_world_size
        self.savepath = savepath

    def find_largest_mask_center(self, mask):
        _, thresholded_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        if largest_contour is None:
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2

        return center_x, center_y

    def crop_image_with_canvas(self, image, center_x, center_y, crop_size):
        h, w = image.shape[:2]
        canvas = np.zeros((crop_size, crop_size), dtype=np.uint16)
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(w, center_x + crop_size // 2)
        y2 = min(h, center_y + crop_size // 2)
        cropped = image[y1:y2, x1:x2]
        start_x = (crop_size - cropped.shape[1]) // 2
        start_y = (crop_size - cropped.shape[0]) // 2
        canvas[start_y:start_y + cropped.shape[0], start_x:start_x + cropped.shape[1]] = cropped
        return canvas

    def process_images(self, spacing_list, png_list, mask_list, filenames):
        image_dict = defaultdict(list)
        os.makedirs(self.savepath, exist_ok=True)
        skipped_masks = []
        for i in tqdm(range(len(spacing_list)), desc="Cropping & Saving"):
            pixel_spacing = spacing_list[i]
            png_img = png_list[i]
            mask_img = mask_list[i]
            filename = filenames[i]

            parts = filename.replace(".png", "").split("_")
            scan_type = parts[0]
            patient_id, study_date = parts[1], parts[2]
            folder_name = patient_id + '_' + study_date

            try:
                crop_size = int(self.real_world_size / pixel_spacing)
            except ValueError as e:
                print(f"Skipping {filename} due to missing ImagerPixelSpacing: {e}")
                continue

            center = self.find_largest_mask_center(mask_img)
            if center is None:
                h, w = png_img.shape[:2]
                center_x, center_y = w // 2, h // 2
                skipped_masks.append(filename)
            else:
                center_x, center_y = center

            cropped_img = self.crop_image_with_canvas(png_img, center_x, center_y, crop_size)

            resized_img = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_AREA)
            
            save_path = os.path.join(self.savepath, folder_name, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, resized_img)
            
            image_dict[folder_name].append((scan_type, resized_img))

        # Ensure AP/LAT order
        for key in image_dict.keys():
            sorted_images = sorted(image_dict[key], key=lambda x: x[0])  # Sort by 'AP' and 'LAT'
            image_dict[key] = [img[1] for img in sorted_images]  # Keep only images
        
        print(f"Processing complete! The cropped image has been saved.")
        if skipped_masks:
            print(f"⚠️  The following {len(skipped_masks)} cases used the IMAGE CENTER because no mask was found:")
            for name in skipped_masks:
                print(f"   - {name}")
        return image_dict

def predict_masks(model, image_tensor, device):
    predicted_masks = []

    with torch.no_grad():
        for input_image in tqdm(image_tensor, desc="Predicting Masks"):
            input_image = input_image.unsqueeze(0).to(device)  # [H,W,C] -> [1,H,W,C]
            output = model(input_image)
            output_np = output.cpu().squeeze().numpy()
            predicted_masks.append(output_np)
            
    return predicted_masks


def run(image_dir=config.INPUT_DIR, save_dir=config.PROCESSED_DIR, gen_dir=config.GENERATED_DIR):
    checkpoint = torch.load(config.MASKING_MODEL_PATH, weights_only=True)
    model = MaskingModel(in_channels=1, out_channels=1).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dicom_processor = DICOMProcessor(image_dir)
    images, original_images, filenames, pixel_spacing = dicom_processor.load_dicom_images()
    image_processor = ImageProcessor(savepath=gen_dir)
    tensor_imgs = image_processor.preprocess_images_to_tensor(images)

    masks = predict_masks(model, tensor_imgs, config.DEVICE)

    resized_masks = image_processor.resize_masks(original_images=original_images, mask_list=masks, filename=filenames)

    mask_processor = MaskProcessor(savepath=save_dir)
    image_dict = mask_processor.process_images(pixel_spacing, original_images, resized_masks, filenames)

    return dict(image_dict)

if __name__ == "__main__":
    tmp = run()