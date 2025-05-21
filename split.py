import os
from pathlib import Path
import shutil
import random

base_dir = Path("D:\CSE-475Project\Handwritten-Amharic-character-Dataset-masters")
output_dir = Path("D:\CSE-475Project\Handwritten-Amharic-character-Dataset-master\split_dataset")

for split in ["train", "val", "test"]:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

image_paths = []
for ext in image_extensions:
    image_paths.extend(base_dir.rglob(f"*{ext}"))

data = []

for img_path in image_paths:
   txt_path = img_path.with_suffix(".txt")
   xml_path = img_path.with_suffix(".xml")
    
   if txt_path.exists():
       data.append((img_path, txt_path))
   elif xml_path.exists():
       data.append((img_path, xml_path))  

print(f"Found {len(data)} image-label pairs.")

random.shuffle(data)
train_split = int(0.7 * len(data))
val_split = int(0.9 * len(data))

train_data = data[:train_split]
val_data = data[train_split:val_split]
test_data = data[val_split:]

def copy_pairs(pairs, split_dir):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    
    for img_file, label_file in pairs:
        rel_path = img_file.relative_to(base_dir).parent

        dest_img_subdir = img_dir / rel_path
        dest_lbl_subdir = lbl_dir / rel_path
        dest_img_subdir.mkdir(parents=True, exist_ok=True)
        dest_lbl_subdir.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_file, dest_img_subdir / img_file.name)
        shutil.copy(label_file, dest_lbl_subdir / label_file.name)

copy_pairs(train_data, output_dir / "train")
copy_pairs(val_data, output_dir / "val")
copy_pairs(test_data, output_dir / "test")

print("\nSplit complete!")
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
print("Files split into 'images' and 'labels' folders under each set.")
