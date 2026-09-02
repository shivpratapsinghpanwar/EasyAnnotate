import os
import shutil
import xml.etree.ElementTree as ET
import cv2

# 1. Ask for the source folder
source_folder = r"C:\Users\shivpratap.panwar\Desktop\safe_house\Door_17july_done\Door_Total"

# 2. Setup the Output Directories
# These will be created in the same parent directory as your source folder
parent_dir = os.path.dirname(source_folder)
sorted_dataset_dir = os.path.join(parent_dir, "Sorted_Dataset_Clean")
visualized_dataset_dir = os.path.join(parent_dir, "Sorted_Dataset_Visualized")

os.makedirs(sorted_dataset_dir, exist_ok=True)
os.makedirs(visualized_dataset_dir, exist_ok=True)

image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

print(f"\nProcessing dataset in: {source_folder}")
print("Creating clean and visualized copies...\n")

# Get all XML files in the source folder
xml_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.xml')]
processed_count = 0

for xml_file in xml_files:
    xml_path = os.path.join(source_folder, xml_file)
    base_name = os.path.splitext(xml_file)[0]
    
    # Find the corresponding image by checking common extensions
    img_path = None
    img_ext_found = None
    for ext in image_exts:
        temp_path = os.path.join(source_folder, base_name + ext)
        # Also check uppercase extensions just in case (e.g., .JPG)
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
        print(f" Warning: Skipping {xml_file} - No matching image found.")
        continue
        
    # Parse the XML to get labels and bounding boxes
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f" Error parsing {xml_file}: {e}")
        continue
    
    boxes = []
    labels = set()
    
    # Standard Pascal VOC format extraction
    for obj in root.findall('object'):
        name_node = obj.find('name')
        if name_node is None:
            continue
            
        name = name_node.text
        labels.add(name)
        
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            # Using int(float()) handles cases where annotation tools save coordinates as decimals
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            boxes.append((name, xmin, ymin, xmax, ymax))
        
    # Determine the folder category based on labels
    if len(labels) == 1:
        category = list(labels)[0]
    elif len(labels) > 1:
        category = "mix"
    else:
        category = "unlabeled_or_empty"
        
    # Create the specific category folders in both output directories
    clean_cat_dir = os.path.join(sorted_dataset_dir, category)
    viz_cat_dir = os.path.join(visualized_dataset_dir, category)
    
    os.makedirs(clean_cat_dir, exist_ok=True)
    os.makedirs(viz_cat_dir, exist_ok=True)
    
    # --- FOLDER 1: The Clean Dataset ---
    # COPY (not move) the original image and XML to keep your source folder intact
    shutil.copy(img_path, os.path.join(clean_cat_dir, base_name + img_ext_found))
    shutil.copy(xml_path, os.path.join(clean_cat_dir, xml_file))
    
    # --- FOLDER 2: The Visualized Dataset ---
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f" Error reading image {img_path} with OpenCV.")
        continue
        
    # Draw every bounding box on the image
    for name, xmin, ymin, xmax, ymax in boxes:
        # Draw a bright green rectangle (OpenCV uses BGR format: Blue, Green, Red)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Add the label text slightly above the top-left corner of the box
        # If the box is too close to the top, push the text inside the box
        text_y = ymin - 10 if ymin - 10 > 10 else ymin + 20
        cv2.putText(img, name, (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    # Save the drawn image to the visualized folder (no XML copied here)
    viz_img_path = os.path.join(viz_cat_dir, base_name + img_ext_found)
    cv2.imwrite(viz_img_path, img)
    
    processed_count += 1

print("-" * 40)
print(f"Dataset QA prep complete! Processed {processed_count} image/xml pairs.")
print(f"Clean Sorted Data: {sorted_dataset_dir}")
print(f"Visualized Data: {visualized_dataset_dir}")