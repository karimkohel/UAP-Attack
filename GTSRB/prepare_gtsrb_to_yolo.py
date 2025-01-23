# prepare_gtsrb_to_yolo.py

import os
import csv
import shutil
import random
from PIL import Image

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

GTSRB_ROOT = "preliminary data/GTSRB_Final_Training"     # Path to your main GTSRB folder
OUTPUT_DIR = "dataset"        # Where we create the images/ and labels/ subfolders
TRAIN_SPLIT = 0.8             # 80% for training, 20% for validation
NUM_CLASSES = 43              # GTSRB has 43 classes

# -------------------------------------------------
def create_yolo_dirs(base_dir):
    """
    Create required directories for YOLO:
      dataset/images/train, dataset/images/val,
      dataset/labels/train, dataset/labels/val
    """
    img_train = os.path.join(base_dir, "images/train")
    img_val   = os.path.join(base_dir, "images/val")
    lbl_train = os.path.join(base_dir, "labels/train")
    lbl_val   = os.path.join(base_dir, "labels/val")
    for d in [img_train, img_val, lbl_train, lbl_val]:
        os.makedirs(d, exist_ok=True)

def convert_bbox_to_yolo(cx1, cy1, cx2, cy2, img_w, img_h):
    """
    Convert bounding box from corner coords (cx1, cy1, cx2, cy2)
    to YOLO format:
      x_center, y_center, width, height (all normalized to [0..1]).
    """
    dw = 1.0 / img_w
    dh = 1.0 / img_h

    x_center = (cx1 + cx2) / 2.0
    y_center = (cy1 + cy2) / 2.0
    w = cx2 - cx1
    h = cy2 - cy1

    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    return x_center, y_center, w, h

def main():
    # 1) Create YOLO directories
    create_yolo_dirs(OUTPUT_DIR)

    # We'll store (image_path, label_str) in a list, then split into train/val
    data_entries = []

    # 2) Iterate over subfolders in GTSRB_ROOT
    # Only keep those that are numeric (like '00000', '00001', etc.)
    subfolders = [f for f in os.listdir(GTSRB_ROOT)
                  if os.path.isdir(os.path.join(GTSRB_ROOT, f)) and f.isdigit()]

    # Sort them numerically (e.g., 00000 -> 00042)
    subfolders.sort()


    for subf in subfolders:
        class_id = int(subf)  # subfolder name is class index, e.g. '00000' -> 0

        csv_name = f"GT-{subf}.csv"  # e.g. GT-00000.csv
        csv_path = os.path.join(GTSRB_ROOT, subf, csv_name)

        if not os.path.isfile(csv_path):
            print(f"Warning: CSV not found in {subf}, skipping.")
            continue

        # Parse CSV (semicolon-delimited)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = csv.reader(f, delimiter=';')
            # Skip header if needed
            header = next(lines, None)
            # Expected columns: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2
            # Example header line might be: "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2"

            for row in lines:
                if len(row) < 7:
                    continue

                filename = row[0].strip()
                img_w = int(row[1])
                img_h = int(row[2])
                x1 = int(row[3])
                y1 = int(row[4])
                x2 = int(row[5])
                y2 = int(row[6])

                image_path = os.path.join(GTSRB_ROOT, subf, filename)
                if not os.path.isfile(image_path):
                    print(f"Warning: Image {filename} not found in {subf}, skipping.")
                    continue

                # Convert to YOLO format
                x_center, y_center, w, h = convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
                label_str = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

                # Store the pair
                data_entries.append((image_path, label_str, class_id))

    print(f"[INFO] Collected {len(data_entries)} bounding boxes total.")

    # 3) Shuffle and split train/val
    random.shuffle(data_entries)

    split_idx = int(TRAIN_SPLIT * len(data_entries))
    train_entries = data_entries[:split_idx]
    val_entries   = data_entries[split_idx:]

    # 4) Copy images into dataset/images/train|val and create label files
    def process_entries(entries, subset):
        img_dir = os.path.join(OUTPUT_DIR, f"images/{subset}")
        lbl_dir = os.path.join(OUTPUT_DIR, f"labels/{subset}")

        for i, (img_src, lbl, class_id) in enumerate(entries):
            basename = os.path.basename(img_src)
            name_no_ext = os.path.splitext(basename)[0]

            # Copy image to YOLO images subfolder
            img_dst = os.path.join(img_dir, f"{class_id}_{basename}")
            if not os.path.exists(img_dst):
                shutil.copy(img_src, img_dst)


            # Write the label text file
            lbl_path = os.path.join(lbl_dir, f"{class_id}_{name_no_ext}.txt")
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(lbl + "\n")

    # Process train/val sets
    process_entries(train_entries, "train")
    process_entries(val_entries,   "val")

    print("[INFO] Data copied & label files created.")

if __name__ == "__main__":
    main()
