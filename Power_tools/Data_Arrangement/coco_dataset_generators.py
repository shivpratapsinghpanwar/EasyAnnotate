import os
import shutil
import json
import xml.etree.ElementTree as ET
from PIL import Image

# 1. Setup paths
source_folder = r"D:\Data_factory\Projects\Project_Fight_NonFight\Fight_person_Data_pascal_voc"
parent_dir = r"D:\Data_factory\Projects\Project_Fight_NonFight\Finished_product\Versions\V1"
coco_dir = os.path.join(parent_dir, "Dataset_COCO")
coco_img_dir = os.path.join(coco_dir, "images")
coco_ann_dir = os.path.join(coco_dir, "annotations")

os.makedirs(coco_img_dir, exist_ok=True)
os.makedirs(coco_ann_dir, exist_ok=True)

image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# 2. COCO JSON Skeleton
coco_data = {
    "info": {"description": "Custom Polygon & BBox Dataset", "version": "1.0"},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

category_mapping = {}
cat_id_counter = 1
ann_id_counter = 1
img_id_counter = 1

print(f"\nProcessing folder: {source_folder}")
print("Building COCO JSON...")

xml_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.xml')]
bg_count, obj_count = 0, 0

# 3. Process every XML
for xml_file in xml_files:
    xml_path = os.path.join(source_folder, xml_file)
    base_name = os.path.splitext(xml_file)[0]
    
    # Match the image
    img_path, img_ext = None, None
    for ext in image_exts:
        for temp_path in [os.path.join(source_folder, base_name + ext), 
                          os.path.join(source_folder, base_name + ext.upper())]:
            if os.path.exists(temp_path):
                img_path, img_ext = temp_path, os.path.splitext(temp_path)[1]
                break
        if img_path: break
            
    if not img_path: continue

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        continue

    # Copy image to COCO folder
    new_img_name = base_name + img_ext
    shutil.copy(img_path, os.path.join(coco_img_dir, new_img_name))

    # Get image size
    size_node = root.find('size')
    if size_node is not None and size_node.find('width') is not None:
        width = int(size_node.find('width').text)
        height = int(size_node.find('height').text)
    else:
        with Image.open(img_path) as im:
            width, height = im.size

    # Add to "images" list
    coco_data["images"].append({
        "id": img_id_counter,
        "file_name": new_img_name,
        "width": width,
        "height": height
    })

    # Find objects (Open, Close, etc.)
    objects = root.findall('object')
    if not objects:
        bg_count += 1  # This is your Hard Negative! No objects found.
    else:
        for obj in objects:
            name = obj.find('name').text
            if name not in category_mapping:
                category_mapping[name] = cat_id_counter
                coco_data["categories"].append({"id": cat_id_counter, "name": name, "supercategory": "none"})
                cat_id_counter += 1

            # Extract Bounding Box
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            
            # Extract Polygon if you used your custom tool
            segmentation = []
            polygon_node = obj.find('polygon')
            if polygon_node is not None:
                poly_points = []
                i = 1
                while True:
                    x_n, y_n = polygon_node.find(f'x{i}'), polygon_node.find(f'y{i}')
                    if x_n is None: break
                    poly_points.extend([int(float(x_n.text)), int(float(y_n.text))])
                    i += 1
                if len(poly_points) >= 6:
                    segmentation = [poly_points]

            # Add to "annotations" list
            coco_data["annotations"].append({
                "id": ann_id_counter,
                "image_id": img_id_counter,
                "category_id": category_mapping[name],
                "bbox": coco_bbox,
                "segmentation": segmentation,
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            })
            ann_id_counter += 1
            obj_count += 1

    img_id_counter += 1

# 4. Save the final JSON
with open(os.path.join(coco_ann_dir, "instances.json"), 'w') as f:
    json.dump(coco_data, f, indent=4)

print("-" * 40)
print(f"Dataset Ready: {coco_dir}")
print(f"Total Images: {img_id_counter - 1}")
print(f"Backgrounds (Hard Negatives): {bg_count}")
print(f"Annotated Objects: {obj_count}")