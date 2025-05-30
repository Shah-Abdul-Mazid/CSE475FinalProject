# # import os
# # import json
# # from PIL import Image
# # import os.path

# # # Category mapping with predefined IDs (class_name: class_id)
# # category_mapping = {
# #     'Bike': 0,
# #     'Bus': 1,
# #     'Car': 2,
# #     'Cng': 3,
# #     'Cycle': 4,
# #     'Mini-Truck': 5,
# #     'People': 6,
# #     'Rickshaw': 7,
# #     'Truck': 8
# # }

# # def yolo_to_coco_json(txt_dir, img_dir, output_json, normalized=True):
# #     # Initialize COCO JSON structure
# #     coco_data = {
# #         "info": {
# #             "description": "Bangladeshi Traffic Flow Dataset - Test Set",
# #             "version": "1.0",
# #             "year": 2025
# #         },
# #         "images": [],
# #         "annotations": [],
# #         "categories": [
# #             {"id": id, "name": name, "supercategory": "vehicle" if name != "People" else "person"}
# #             for name, id in category_mapping.items()
# #         ]
# #     }
    
# #     image_ids = {}
# #     ann_id = 1
    
# #     txt_files = []
# #     for root, _, files in os.walk(txt_dir):
# #         for file in files:
# #             if file.endswith('.txt'):
# #                 txt_files.append(os.path.join(root, file))
    
# #     if not txt_files:
# #         raise FileNotFoundError(f"No .txt files found in {txt_dir} or its subdirectories")
    
# #     id_to_name = {v: k for k, v in category_mapping.items()}
    
# #     for txt_file in txt_files:
# #         base_name = os.path.splitext(os.path.basename(txt_file))[0]
# #         rel_path = os.path.relpath(os.path.dirname(txt_file), txt_dir)
# #         img_file = os.path.join(img_dir, rel_path, f"{base_name}.jpg")
        
# #         if not os.path.exists(img_file):
# #             print(f"Warning: Image {img_file} not found, skipping {txt_file}")
# #             continue
        
# #         # Get image dimensions
# #         try:
# #             with Image.open(img_file) as img:
# #                 img_width, img_height = img.size
# #         except Exception as e:
# #             print(f"Warning: Could not read {img_file}, skipping. Error: {e}")
# #             continue
        
# #         # Add image to COCO JSON
# #         image_id = len(image_ids) + 1
# #         image_key = f"{rel_path}/{base_name}".replace("\\", "/")
# #         image_ids[image_key] = image_id
# #         rel_img_path = os.path.join("test/images", rel_path, f"{base_name}.jpg").replace("\\", "/")
# #         coco_data["images"].append({
# #             "id": image_id,
# #             "file_name": rel_img_path,
# #             "width": img_width,
# #             "height": img_height,
# #             "license": None,
# #             "date_captured": None
# #         })
        
# #         # Read annotations from .txt file
# #         with open(txt_file, 'r') as f:
# #             for line in f:
# #                 line = line.strip()
# #                 if not line:
# #                     continue
                
# #                 try:
# #                     parts = line.split()
# #                     if len(parts) < 5:
# #                         print(f"Skipping invalid line in {txt_file}: {line}")
# #                         continue
                    
# #                     class_id, x_center, y_center, width, height = map(float, parts[:5])
# #                     class_id = int(class_id)
                    
# #                     if class_id not in id_to_name:
# #                         print(f"Warning: Unknown class_id '{class_id}' in {txt_file}, skipping")
# #                         continue
                    
# #                     if normalized:
# #                         x_center *= img_width
# #                         y_center *= img_height
# #                         width *= img_width
# #                         height *= img_height
                    
# #                     x_min = x_center - width / 2
# #                     y_min = y_center - height / 2
                    
# #                     coco_data["annotations"].append({
# #                         "id": ann_id,
# #                         "image_id": image_ids[image_key],
# #                         "category_id": class_id,
# #                         "bbox": [round(x_min, 2), round(y_min, 2), round(width, 2), round(height, 2)],
# #                         "area": round(width * height, 2),
# #                         "iscrowd": 0,
# #                         "segmentation": []
# #                     })
# #                     ann_id += 1
# #                 except ValueError as e:
# #                     print(f"Skipping invalid line in {txt_file}: {line}. Error: {e}")
# #                     continue
    
# #     with open(output_json, 'w') as json_file:
# #         json.dump(coco_data, json_file, indent=4)
    
# #     print(f"Converted {len(txt_files)} text files to {output_json}")
# #     return coco_data

# # # Corrected directory paths
# # test_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\labels"
# # test_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\images"

# # # train_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\labels"
# # # train_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\images"

# # # val_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\labels"
# # # val_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\images"

# # test_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\test.json"
# # # train_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\train.json"
# # # val_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\val.json"

# # # Run conversion
# # test_json_data = yolo_to_coco_json(test_txt_dir, test_img_dir, test_output_json, normalized=True)


# # # Print sample of each JSON
# # print("Test JSON sample:")
# # print(json.dumps(test_json_data, indent=4)[:1000], "... (truncated)")
# # # print("\nTrain JSON sample:")
# # # print(json.dumps(train_json_data, indent=4)[:1000], "... (truncated)")
# # # print("\nVal JSON sample:")
# # # print(json.dumps(val_json_data, indent=4)[:1000], "... (truncated)")

# import os
# import json
# from PIL import Image
# import os.path

# # Category mapping with predefined IDs (class_name: class_id)
# category_mapping = {
#     'Bike': 0,
#     'Bus': 1,
#     'Car': 2,
#     'Cng': 3,
#     'Cycle': 4,
#     'Mini-Truck': 5,
#     'People': 6,
#     'Rickshaw': 7,
#     'Truck': 8
# }

# def yolo_to_coco_json(txt_dir, img_dir, output_json, normalized=True):
#     # Initialize COCO JSON structure
#     coco_data = {
#         "info": {
#             "description": "Bangladeshi Traffic Flow Dataset - Test Set",
#             "version": "1.0",
#             "year": 2025
#         },
#         "images": [],
#         "annotations": [],
#         "categories": [
#             {"id": id, "name": name, "supercategory": "vehicle" if name != "People" else "person"}
#             for name, id in category_mapping.items()
#         ]
#     }
    
#     image_ids = {}
#     ann_id = 1
    
#     txt_files = []
#     for root, _, files in os.walk(txt_dir):
#         for file in files:
#             if file.endswith('.txt'):
#                 txt_files.append(os.path.join(root, file))
    
#     if not txt_files:
#         raise FileNotFoundError(f"No .txt files found in {txt_dir} or its subdirectories")
    
#     id_to_name = {v: k for k, v in category_mapping.items()}
    
#     for txt_file in txt_files:
#         base_name = os.path.splitext(os.path.basename(txt_file))[0]
#         rel_path = os.path.relpath(os.path.dirname(txt_file), txt_dir)
#         img_file = os.path.join(img_dir, rel_path, f"{base_name}.jpg")
        
#         if not os.path.exists(img_file):
#             print(f"Warning: Image {img_file} not found, skipping {txt_file}")
#             continue
        
#         # Get image dimensions
#         try:
#             with Image.open(img_file) as img:
#                 img_width, img_height = img.size
#         except Exception as e:
#             print(f"Warning: Could not read {img_file}, skipping. Error: {e}")
#             continue
        
#         image_id = len(image_ids) + 1
#         image_key = f"{rel_path}/{base_name}".replace("\\", "/")
#         image_ids[image_key] = image_id
#         rel_img_path = os.path.join("test/images", rel_path, f"{base_name}.jpg").replace("\\", "/")
#         coco_data["images"].append({
#             "id": image_id,
#             "file_name": rel_img_path,
#             "width": img_width,
#             "height": img_height,
#             "license": None,
#             "date_captured": None
#         })
        
#         # Read annotations from .txt file
#         with open(txt_file, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
                
#                 try:
#                     parts = line.split()
#                     if len(parts) < 5:
#                         print(f"Skipping invalid line in {txt_file}: {line}")
#                         continue
                    
#                     class_id, x_center, y_center, width, height = map(float, parts[:5])
#                     class_id = int(class_id)
                    
#                     if class_id not in id_to_name:
#                         print(f"Warning: Unknown class_id '{class_id}' in {txt_file}, skipping")
#                         continue
                    
#                     if normalized:
#                         x_center *= img_width
#                         y_center *= img_height
#                         width *= img_width
#                         height *= img_height
                    
#                     x_min = x_center - width / 2
#                     y_min = y_center - height / 2
                    
#                     coco_data["annotations"].append({
#                         "id": ann_id,
#                         "image_id": image_ids[image_key],
#                         "category_id": class_id,
#                         "bbox": [round(x_min, 2), round(y_min, 2), round(width, 2), round(height, 2)],
#                         "area": round(width * height, 2),
#                         "iscrowd": 0,
#                         "segmentation": []
#                     })
#                     ann_id += 1
#                 except ValueError as e:
#                     print(f"Skipping invalid line in {txt_file}: {line}. Error: {e}")
#                     continue
    
#     with open(output_json, 'w') as json_file:
#         json.dump(coco_data, json_file, indent=4)
    
#     print(f"Converted {len(txt_files)} text files to {output_json}")
#     return coco_data

# # Directory paths for test set
# test_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\labels"
# test_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\images"
# test_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\test.json"

# train_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\labels"
# train_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\images"
# train_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\train.json"

# val_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\labels"
# val_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\images"
# val_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\val.json"

# # Run conversion for test set
# test_json_data = yolo_to_coco_json(test_txt_dir, test_img_dir, test_output_json, normalized=True)
# train_json_data = yolo_to_coco_json(train_txt_dir, train_img_dir, train_output_json, normalized=True)
# val_json_data = yolo_to_coco_json(val_txt_dir, val_img_dir, val_output_json, normalized=True)
import os
import json
from PIL import Image
import os.path

# Category mapping with predefined IDs (class_name: class_id)
category_mapping = {
    'Bike': 0,
    'Bus': 1,
    'Car': 2,
    'Cng': 3,
    'Cycle': 4,
    'Mini-Truck': 5,
    'People': 6,
    'Rickshaw': 7,
    'Truck': 8
}

def yolo_to_coco_json(txt_dir, img_dir, output_json, normalized=True):
    # Determine dataset split from txt_dir or output_json (e.g., 'test', 'train', 'val')
    split_name = os.path.basename(os.path.dirname(txt_dir))  # Gets 'test', 'train', or 'val'
    description = f"Bangladeshi Traffic Flow Dataset - {split_name.capitalize()} Set"

    # Initialize COCO JSON structure
    coco_data = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2025
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": id, "name": name, "supercategory": "vehicle" if name != "People" else "person"}
            for name, id in category_mapping.items()
        ]
    }
    
    image_ids = {}
    ann_id = 1
    
    txt_files = []
    for root, _, files in os.walk(txt_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {txt_dir} or its subdirectories")
    
    id_to_name = {v: k for k, v in category_mapping.items()}
    
    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        rel_path = os.path.relpath(os.path.dirname(txt_file), txt_dir)
        img_file = os.path.join(img_dir, rel_path, f"{base_name}.jpg")
        
        if not os.path.exists(img_file):
            print(f"Warning: Image {img_file} not found, skipping {txt_file}")
            continue
        
        # Get image dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Warning: Could not read {img_file}, skipping. Error: {e}")
            continue
        
        # Add image to COCO JSON
        image_id = len(image_ids) + 1
        image_key = f"{rel_path}/{base_name}".replace("\\", "/")
        image_ids[image_key] = image_id
        # Use dynamic split_name for rel_img_path
        rel_img_path = os.path.join(f"{split_name}/images", rel_path, f"{base_name}.jpg").replace("\\", "/")
        coco_data["images"].append({
            "id": image_id,
            "file_name": rel_img_path,
            "width": img_width,
            "height": img_height,
            "license": None,
            "date_captured": None
        })
        
        # Read annotations from .txt file
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) < 5:
                        print(f"Skipping invalid line in {txt_file}: {line}")
                        continue
                    
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    class_id = int(class_id)
                    
                    if class_id not in id_to_name:
                        print(f"Warning: Unknown class_id '{class_id}' in {txt_file}, skipping")
                        continue
                    
                    if normalized:
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height
                    
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": image_ids[image_key],
                        "category_id": class_id,
                        "bbox": [round(x_min, 2), round(y_min, 2), round(width, 2), round(height, 2)],
                        "area": round(width * height, 2),
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id += 1
                except ValueError as e:
                    print(f"Skipping invalid line in {txt_file}: {line}. Error: {e}")
                    continue
    
    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)
    
    print(f"Converted {len(txt_files)} text files to {output_json}")
    return coco_data

# Directory paths for test, train, and val sets
test_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\labels"
test_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\images"
test_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\test.json"

train_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\labels"
train_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\images"
train_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\train.json"

val_txt_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\labels"
val_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\images"
val_output_json = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\val.json"

# Run conversion for all sets
test_json_data = yolo_to_coco_json(test_txt_dir, test_img_dir, test_output_json, normalized=True)
train_json_data = yolo_to_coco_json(train_txt_dir, train_img_dir, train_output_json, normalized=True)
val_json_data = yolo_to_coco_json(val_txt_dir, val_img_dir, val_output_json, normalized=True)

# Print sample of each JSON
print("Test JSON sample:")
print(json.dumps(test_json_data, indent=4)[:1000], "... (truncated)")
print("\nTrain JSON sample:")
print(json.dumps(train_json_data, indent=4)[:1000], "... (truncated)")
print("\nVal JSON sample:")
print(json.dumps(val_json_data, indent=4)[:1000], "... (truncated)")