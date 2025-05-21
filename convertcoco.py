import os
import json
from PIL import Image
from tqdm import tqdm

class_map = {
    "Bike": 1, "Bus": 2, "Car": 3, "Cng": 4, "Cycle": 5,
    "Mini-Truck": 6, "People": 7, "Rickshaw": 8, "Truck": 9
}

def yolo_to_coco(img_root, label_root, output_json, class_map, starting_image_id=1, starting_ann_id=1):
    images = []
    annotations = []
    annotation_id = starting_ann_id
    image_id = starting_image_id

    for root, _, files in os.walk(img_root):
        for file in tqdm(files, desc=f"Processing {os.path.basename(img_root)}"):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(img_path, img_root).replace("\\", "/")
            label_path = os.path.join(label_root, rel_path).replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")

            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for {img_path}")
                continue

            try:
                img = Image.open(img_path)
                width, height = img.size
            except Exception as e:
                print(f"Warning: Could not read image {img_path}: {e}")
                continue

            images.append({
                "id": image_id,
                "file_name": rel_path,
                "width": width,
                "height": height
            })

            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid label format in {label_path}: {line.strip()}")
                        continue
                    class_id, x_center, y_center, w, h = map(float, parts)
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id) + 1,
                        "bbox": [x_min, y_min, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            image_id += 1

    categories = [{"id": v, "name": k} for k, v in class_map.items()]
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=2)

    print(f"✅ Saved COCO JSON: {output_json}")
    return image_id, annotation_id

base_path = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset"
save_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\annotations"

os.makedirs(save_dir, exist_ok=True)

splits = ["train", "val", "test"]
start_img_id = 1
start_ann_id = 1

for split in splits:
    img_dir = os.path.join(base_path, split, "images")
    label_dir = os.path.join(base_path, split, "labels")
    output_json = os.path.join(save_dir, f"{split}_annotations_coco.json")

    start_img_id, start_ann_id = yolo_to_coco(
        img_root=img_dir,
        label_root=label_dir,
        output_json=output_json,
        class_map=class_map,
        starting_image_id=start_img_id,
        starting_ann_id=start_ann_id
    )