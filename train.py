# train_surrogate_yolo.py

import os
from ultralytics import YOLO


def main():
    data_yaml = "dataset/gtsrb.yaml"  # The file created by the data preprocessing script
    model_arch = "yolo11s.pt"  # You can choose yolov8s.yaml, yolov8m.yaml, etc. for larger models
    
    # 1) Create YOLO object
    model = YOLO(model_arch)  # from ultralytics, loads a YOLO model
    
    # 2) Train the model
    model.train(
        data=data_yaml,   # path to our GTSRB YAML
        imgsz=640,        # can adjust image size
        epochs=10,        # for demonstration, might need more
        batch=8,          # adjust based on your GPU memory
        name="gtsrb_surrogate",  # experiment name
        amp=False,        # mixed precision training
        verbose=True
    )

if __name__ == "__main__":
    main()
