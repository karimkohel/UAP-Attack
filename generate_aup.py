# generate_uap_yolo.py
import os
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import glob

###########################
# CONFIG
###########################
CHECKPOINT_PATH = "runs/detect/gtsrb_surrogate/weights/best.pt"  # or wherever your trained YOLO checkpoint is
# IMAGE_FOLDER = "dataset/images/train"  # We'll use the training set as the representative set
IMAGE_FOLDER = "uapImages"  # We'll use the training set as the representative set
IMG_SIZE = 640   # YOLO inference size
EPSILON = 8/255  # L-inf limit (~0.031 in [0,1] scale)
ALPHA = 1/255    # step size (~0.0039 in [0,1])
EPOCHS = 2       # how many times we pass over the images
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UAP_SAVE_PATH = "universal_perturbation_u8.pt"

###########################
# DATASET
###########################
class YoloImageDataset(Dataset):
    """
    A simple dataset that loads images from a folder (jpg|png|ppm).
    No labels needed for this purpose; we rely on YOLO's internal training step 
    or we can just run forward on the model. We'll artificially create a 'dummy' label for the forward pass.
    """
    def __init__(self, img_dir, img_size=640):
        self.img_paths = []
        exts = ["*.jpg", "*.png", "*.ppm", "*.jpeg"]
        for e in exts:
            self.img_paths += glob.glob(os.path.join(img_dir, e))
        self.img_paths.sort()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        # load image in BGR
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # just in case
            raise ValueError(f"Could not read {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # resize
        img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size))
        # to tensor, range [0,1]
        tensor = self.transform(img_rgb)  # shape [3, 640, 640]
        return tensor, path


def clamp_uap(uap, eps):
    return torch.clamp(uap, min=-eps, max=eps)

def main():
    # 1) Load the YOLO model in 'train' mode so we can access loss
    model = YOLO(CHECKPOINT_PATH)
    model.to(DEVICE)

    # 2) Build dataset & dataloader
    dataset = YoloImageDataset(IMAGE_FOLDER, IMG_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 3) Initialize UAP
    # shape [3, 640, 640]
    uap = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
    
    # 4) Loop over multiple epochs
    for epoch in range(EPOCHS):
        print(f"=== UAP Epoch {epoch+1}/{EPOCHS} ===")
        for i, (img_tensor, _) in enumerate(loader):
            img_tensor = img_tensor.to(DEVICE)  # shape [1,3,H,W]
            
            # Make the image require grad
            adv_img = (img_tensor + uap).clone().detach().requires_grad_(True)

            preds = model.model(adv_img)  # list of [batch, num_anchors, (cx,cy,w,h,conf,cls...)]

            # Combine all predictions into one tensor:
            total_loss = 0
            if isinstance(preds, (list, tuple)):
                for p in preds:
                    if p is not None and isinstance(p, torch.Tensor):
                        total_loss += p.abs().sum()
            else:
                # If not list or tuple, treat as single output
                total_loss += preds.abs().sum()
            
            # We want to MAXIMIZE this "loss" (to degrade the detection),
            # so we do gradient ascent. We'll do: loss = - total_loss for backprop minimization.
            loss = -total_loss  # negative sign => gradient ascent
            loss.backward()

            # Now adv_img.grad has the gradient we want
            grad_sign = adv_img.grad.detach().sign()
            # Accumulate a small step in the direction of grad
            uap = uap + ALPHA * grad_sign
            # Project
            uap = clamp_uap(uap, EPSILON)

            # optional logging
            if i % 100 == 0:
                print(f"  Processed {i} images for epoch {epoch+1}, current total_loss={total_loss.item():.4f}")

    print("=== UAP generation complete ===")
    # 5) Save UAP
    torch.save(uap, UAP_SAVE_PATH)
    print(f"UAP saved to {UAP_SAVE_PATH}")

if __name__ == "__main__":
    main()
