import argparse
import os
import cv2
import xml.etree.ElementTree as ET

# --- Configuration ---
parser = argparse.ArgumentParser(description="Draw Pascal VOC boxes onto copies of the images.")
parser.add_argument("source_folders", nargs="+", help="folders of images with their XMLs (not modified)")
parser.add_argument("--output", required=True, help="where the drawn images go, one subfolder per source")
args = parser.parse_args()

# Your original source paths (These will NOT be modified)
source_folders = {os.path.basename(os.path.normpath(p)): p for p in args.source_folders}

# The new location where visualized images will be saved
output_base = args.output
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
box_color = (0, 255, 0)  # Bright Green
thickness = 2

# Create the main output directory
os.makedirs(output_base, exist_ok=True)

print(" Starting Non-Destructive Visualization...")

for folder_label, src_path in source_folders.items():
    if not os.path.exists(src_path):
        print(f" Path not found: {src_path}")
        continue
    
    # Create the specific subfolder in the new location
    dest_subfolder = os.path.join(output_base, folder_label)
    os.makedirs(dest_subfolder, exist_ok=True)
    
    print(f"Processing: {folder_label}")
    
    # Iterate through XML files in the source
    xml_files = [f for f in os.listdir(src_path) if f.lower().endswith('.xml')]
    
    for xml_file in xml_files:
        xml_full_path = os.path.join(src_path, xml_file)
        base_name = os.path.splitext(xml_file)[0]
        
        # Find the matching image in the source folder
        img_src_path = None
        img_ext = None
        for ext in image_extensions:
            potential_img = os.path.join(src_path, f"{base_name}{ext}")
            if os.path.exists(potential_img):
                img_src_path = potential_img
                img_ext = ext
                break
        
        if img_src_path:
            try:
                # 1. Read the original image
                image = cv2.imread(img_src_path)
                if image is None: continue
                
                # 2. Parse XML for coordinates
                tree = ET.parse(xml_full_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    bbox = obj.find('bndbox')
                    
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    
                    # 3. Draw the Bounding Box on the image copy in memory
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, thickness)
                    cv2.putText(image, label, (xmin, ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # 4. Save the new image to the NEW folder (same name as original)
                new_img_name = f"{base_name}{img_ext}"
                cv2.imwrite(os.path.join(dest_subfolder, new_img_name), image)
                
            except Exception as e:
                print(f" Error processing {base_name}: {e}")

print("-" * 30)
print(f" Finished! Check your new images here: {output_base}")