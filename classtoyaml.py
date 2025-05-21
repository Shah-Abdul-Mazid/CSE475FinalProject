import yaml
import xml.etree.ElementTree as ET
from pathlib import Path

base_txt_path = Path("Bd_Traffic_Dataset_v6/Bangladeshi Traffic Flow Dataset/Bangladeshi Traffic Flow Dataset/split_dataset")
base_xml_path = Path("Bd_Traffic_Dataset_v6/Bangladeshi Traffic Flow Dataset/Bangladeshi Traffic Flow Dataset/Annotated Images")
yaml_output_path = Path("Bd_Traffic_Dataset_v6/Bangladeshi Traffic Flow Dataset/Bangladeshi Traffic Flow Dataset/data.yaml")

unique_classes = set()

for txt_file in base_txt_path.rglob("*.txt"):
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0].isdigit():  # Only class IDs
                    unique_classes.add(int(parts[0]))
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")

class_names = set()
for xml_file in base_xml_path.rglob("*.xml"):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            class_names.add(name)
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")

class_list = sorted(class_names)

yaml_output_path.parent.mkdir(parents=True, exist_ok=True)

with open(yaml_output_path, 'w', encoding='utf-8') as f:
    f.write("train: Bd_Traffic_Dataset_v6/Bangladeshi Traffic Flow Dataset/Bangladeshi Traffic Flow Dataset/split_dataset/train/images\n")
    f.write("val: Bd_Traffic_Dataset_v6/Bangladeshi Traffic Flow Dataset/Bangladeshi Traffic Flow Dataset/split_dataset/val/images\n")
    f.write("test: Bd_Traffic_Dataset_v6/Bangladeshi Traffic Flow Dataset/Bangladeshi Traffic Flow Dataset/split_dataset/test/images\n")
    f.write(f"nc: {len(class_list)}\n")
    f.write("names: [ " + ", ".join(f"'{name}'" for name in class_list) + " ]\n")

print(f"\ndata.yaml saved successfully at {yaml_output_path}")
print("Class Names:")
for idx, name in enumerate(class_list):
    print(f"  {idx}: {name}")
