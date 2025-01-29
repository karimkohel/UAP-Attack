# apply_uap.py

import os
import argparse
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

#  python apply_uap.py --image path/to/any_image.jpg

UAP_PATH = "universal_perturbation_u8.pt"
YOLO_CHECKPOINT = "runs/detect/gtsrb_surrogate/weights/best.pt"  # "Surrogate" checkpoint
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_uap(uap_path):
    return torch.load(uap_path, map_location=DEVICE)

def preprocess_image(img_bgr, size=640):
    img_bgr = cv2.resize(img_bgr, (size, size))
    tensor = torch.from_numpy(img_bgr).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # shape [1,3,H,W]
    return tensor

def visualize_results(model, original_tensor, adv_tensor):

    orig_results = model.predict(original_tensor, conf=0.25)
    adv_results  = model.predict(adv_tensor, conf=0.25)

    # For single images, each is just [0]
    orig_res = orig_results[0]
    adv_res  = adv_results[0]

    orig_plot_bgr = orig_res.plot()
    adv_plot_bgr  = adv_res.plot()

    # Convert BGR -> RGB for display
    orig_plot_rgb = cv2.cvtColor(orig_plot_bgr, cv2.COLOR_BGR2RGB)
    adv_plot_rgb  = cv2.cvtColor(adv_plot_bgr, cv2.COLOR_BGR2RGB)

    # Show side-by-side in Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(orig_plot_rgb)
    axs[0].set_title("Original Detection")
    axs[0].axis("off")

    axs[1].imshow(adv_plot_rgb)
    axs[1].set_title("Adversarial Detection")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to a test image")
    args = parser.parse_args()

    # 1) Load YOLO model + UAP
    model = YOLO(YOLO_CHECKPOINT)
    model.to(DEVICE)
    uap = load_uap(UAP_PATH)  # shape [1,3,640,640]

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise ValueError(f"Could not load image {args.image}")

    # 3) Preprocess: keep BGR => [1,3,H,W] in [0..1]
    orig_tensor = preprocess_image(img_bgr, IMG_SIZE).to(DEVICE)

    # 4) Create adversarial version (clip to [0,1])
    adv_tensor = torch.clamp(orig_tensor + uap, 0, 1)

    visualize_results(model, orig_tensor, adv_tensor)

if __name__ == "__main__":
    main()