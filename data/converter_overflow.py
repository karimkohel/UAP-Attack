import os
import pandas as pd
import cv2

# Define paths
dataset_path = "data/GTSRB_train/Final_Test/"  # Change to your dataset folder
csv_file = os.path.join(dataset_path, "GT-final_test.test.csv")  # Test CSV file
images_dir = os.path.join(dataset_path, "Images")  # Test images folder
yolo_labels_dir = os.path.join(dataset_path, "labels")  # YOLO labels output directory

# Create labels directory
os.makedirs(yolo_labels_dir, exist_ok=True)

# Load GTSRB CSV file
df = pd.read_csv(csv_file, sep=";")  # GTSRB uses ';' as separator

# Convert annotations to YOLO format
for _, row in df.iterrows():
    filename = row["Filename"]
    width, height = row["Width"], row["Height"]
    x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
    class_id = row["ClassId"]

    # Convert to YOLO format (normalize)
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    bbox_width = (x2 - x1) / width
    bbox_height = (y2 - y1) / height

    # Save as text file
    yolo_filename = os.path.join(yolo_labels_dir, filename.replace(".ppm", ".txt"))
    with open(yolo_filename, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print("✅ GTSRB annotations converted to YOLO format!")
