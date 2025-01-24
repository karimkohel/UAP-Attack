# apply_uap.py

import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

"""
Loads the final UAP (universal_perturbation_u8.pt) and applies it to new images
to see how the YOLO detection is affected.

Usage:
  python apply_uap.py --image path/to/any_image.jpg
"""

UAP_PATH = "universal_perturbation_u8.pt"
YOLO_CHECKPOINT = "runs/detect/gtsrb_surrogate/weights/best.pt"  # Surrogate checkpoint
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_uap(uap_path):
    return torch.load(uap_path, map_location=DEVICE)

def preprocess_image(img_bgr, size=640):
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize
    img_rgb = cv2.resize(img_rgb, (size, size))
    # Convert to float tensor [0..1]
    tensor = torch.from_numpy(img_rgb).float() / 255.0
    # Rearrange to (C,H,W)
    tensor = tensor.permute(2,0,1).unsqueeze(0)
    return tensor

def postprocess_tensor(tensor):
    # Convert back to [0..255], HWC, BGR for visualization if needed
    img = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    img = np.clip(img, 0, 1) * 255
    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img_bgr

def visualize_results(model, original_tensor, adv_tensor):
    """
    Runs YOLO detection on both original and adv images, 
    then shows bounding boxes side-by-side for comparison.
    """
    # Predictions
    orig_preds = model.predict(original_tensor, conf=0.25)
    adv_preds  = model.predict(adv_tensor, conf=0.25)

    # Convert to CV2 images
    orig_img_bgr = postprocess_tensor(original_tensor)
    adv_img_bgr = postprocess_tensor(adv_tensor)
    
    # Draw YOLO results (Ultralytics result has .boxes)
    # For demonstration, we just display the final annotated images from YOLO's .plot()
    orig_plot = orig_preds[0].plot()
    adv_plot = adv_preds[0].plot()

    # Show side by side
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(cv2.cvtColor(orig_plot, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Detection")
    axs[0].axis('off')
    axs[1].imshow(cv2.cvtColor(adv_plot, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Adversarial Detection")
    axs[1].axis('off')
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to a test image")
    args = parser.parse_args()
    
    # 1) Load YOLO model and UAP
    model = YOLO(YOLO_CHECKPOINT)
    model.to(DEVICE)
    uap = load_uap(UAP_PATH)  # shape [1,3,640,640]
    
    # 2) Load and preprocess the input image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise ValueError(f"Could not load image {args.image}")
    
    orig_tensor = preprocess_image(img_bgr, IMG_SIZE).to(DEVICE)
    
    # 3) Create adversarial version
    adv_tensor = torch.clamp(orig_tensor + uap, 0, 1)
    
    # 4) Visualize the results
    visualize_results(model, orig_tensor, adv_tensor)

if __name__ == "__main__":
    main()
