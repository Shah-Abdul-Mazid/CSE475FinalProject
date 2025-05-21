import xml.etree.ElementTree as ET
from pathlib import Path

base_dir = Path("D:\\CSE-475Project\\Bd_Traffic_Dataset_v6\\Bangladeshi Traffic Flow Dataset\\Bangladeshi Traffic Flow Dataset\\Annotated Images")

xml_files = list(base_dir.rglob("*.xml"))

def build_class_map(xml_files):
    class_set = set()
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                class_set.add(obj.find("name").text.strip())
        except Exception as e:
            print(f"⚠️ Error parsing {xml_file.name}: {e}")
    return {cls: i for i, cls in enumerate(sorted(class_set))}

def convert_xml_to_yolo(xml_file, class_map):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        txt_file = xml_file.with_suffix(".txt")
        with open(txt_file, "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text.strip()
                class_id = class_map[class_name]

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        print(f"{xml_file.relative_to(base_dir)} ➝ {txt_file.name}")
    except Exception as e:
        print(f"Failed to convert {xml_file.name}: {e}")

if __name__ == "__main__":
    if not xml_files:
        print("No XML files found.")
        exit()

    class_map = build_class_map(xml_files)

    print("\nDetected Classes:")
    for name, idx in class_map.items():
        print(f"  {idx}: {name}")

    for xml_file in xml_files:
        convert_xml_to_yolo(xml_file, class_map)

    print("\nAll files converted!")
