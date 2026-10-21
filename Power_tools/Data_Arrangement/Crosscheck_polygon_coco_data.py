import argparse
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

# 1. Ask for the source folder
parser = argparse.ArgumentParser(description="Draw polygons or boxes from Pascal VOC XMLs into two review folders.")
parser.add_argument("source_folder", help="folder of images with their Pascal VOC XMLs")
source_folder = parser.parse_args().source_folder

# 2. Setup the two Output Directories
parent_dir = os.path.dirname(source_folder)
dir_poly = os.path.join(parent_dir, "Polygons_Visualized")
dir_bbox = os.path.join(parent_dir, "BBox_Only_Visualized")

os.makedirs(dir_poly, exist_ok=True)
os.makedirs(dir_bbox, exist_ok=True)

image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

print(f"\nScanning hybrid dataset in: {source_folder}")
print("Separating into two folders (Polygons vs BBoxes)...\n")

xml_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.xml')]

count_poly = 0
count_bbox = 0

for xml_file in xml_files:
    xml_path = os.path.join(source_folder, xml_file)
    base_name = os.path.splitext(xml_file)[0]
    
    # Find the corresponding image
    img_path = None
    img_ext_found = None
    for ext in image_exts:
        temp_path = os.path.join(source_folder, base_name + ext)
        temp_path_upper = os.path.join(source_folder, base_name + ext.upper())
        
        if os.path.exists(temp_path):
            img_path = temp_path
            img_ext_found = ext
            break
        elif os.path.exists(temp_path_upper):
            img_path = temp_path_upper
            img_ext_found = ext.upper()
            break
            
    if not img_path:
        continue
        
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f" Error parsing {xml_file}: {e}")
        continue
        
    img = cv2.imread(img_path)
    if img is None:
        continue

    # 1st Pass: Determine what kind of file this is
    is_polygon_file = False
    for obj in root.findall('object'):
        if obj.find('polygon') is not None:
            is_polygon_file = True
            break
            
    has_objects = False
    
    # 2nd Pass: Draw the appropriate shapes
    for obj in root.findall('object'):
        name_node = obj.find('name')
        if name_node is None:
            continue
            
        name = name_node.text
        has_objects = True
        
        if is_polygon_file:
            # File has polygon data, so we draw polygons
            polygon_node = obj.find('polygon')
            if polygon_node is not None:
                points = []
                i = 1
                while True:
                    x_node = polygon_node.find(f'x{i}')
                    y_node = polygon_node.find(f'y{i}')
                    if x_node is None or y_node is None:
                        break
                    x = int(float(x_node.text))
                    y = int(float(y_node.text))
                    points.append([x, y])
                    i += 1
                
                if len(points) >= 3:
                    pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)
                    text_x, text_y = points[0][0], points[0][1]
                    cv2.putText(img, name, (text_x, max(text_y - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        else:
            # Old file with no polygon data, so we draw standard Bounding Boxes
            bndbox_node = obj.find('bndbox')
            if bndbox_node is not None:
                xmin = int(float(bndbox_node.find('xmin').text))
                ymin = int(float(bndbox_node.find('ymin').text))
                xmax = int(float(bndbox_node.find('xmax').text))
                ymax = int(float(bndbox_node.find('ymax').text))
                
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(img, name, (xmin, max(ymin - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save to the correct folder
    if has_objects:
        output_filename = base_name + img_ext_found
        if is_polygon_file:
            cv2.imwrite(os.path.join(dir_poly, output_filename), img)
            count_poly += 1
        else:
            cv2.imwrite(os.path.join(dir_bbox, output_filename), img)
            count_bbox += 1

print("-" * 40)
print(f"Done! Successfully separated the dataset.")
print(f"  - Images routed to Polygons_Visualized: {count_poly}")
print(f"  - Images routed to BBox_Only_Visualized: {count_bbox}")
print("-" * 40)