import argparse
import os
import shutil
import xml.etree.ElementTree as ET

# --- Configuration ---
parser = argparse.ArgumentParser(description="Move orphaned images/XMLs aside and rename labels in the rest.")
parser.add_argument("target_folder", help="folder of images and Pascal VOC XMLs")
parser.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                    help="label to rename; repeat for several")
args = parser.parse_args()
target_folder = args.target_folder

# Label Mapping: "Old Label": "New Label"
label_map = dict(pair.split("=", 1) for pair in args.rename)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Setup Orphan Folders
orphan_root = os.path.join(target_folder, "cleanup_orphans")
orphan_images_dir = os.path.join(orphan_root, "images_without_xml")
orphan_xml_dir = os.path.join(orphan_root, "xml_without_images")

os.makedirs(orphan_images_dir, exist_ok=True)
os.makedirs(orphan_xml_dir, exist_ok=True)

print(f" Processing labels and cross-checking pairs in: {target_folder}")

# 1. Identify all files and separate them by extension
all_files = os.listdir(target_folder)
image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
xml_files = [f for f in all_files if f.lower().endswith('.xml')]

# Create sets of base names (filenames without extensions) for fast lookup
image_basenames = {os.path.splitext(f)[0] for f in image_files}
xml_basenames = {os.path.splitext(f)[0] for f in xml_files}

# 2. Handle Orphaned Images (Image exists, but no XML)
orphaned_images_count = 0
for img_file in image_files:
    base_name = os.path.splitext(img_file)[0]
    if base_name not in xml_basenames:
        shutil.move(os.path.join(target_folder, img_file), os.path.join(orphan_images_dir, img_file))
        orphaned_images_count += 1

# 3. Handle Orphaned XMLs (XML exists, but no Image)
orphaned_xml_count = 0
for xml_file in xml_files:
    base_name = os.path.splitext(xml_file)[0]
    if base_name not in image_basenames:
        shutil.move(os.path.join(target_folder, xml_file), os.path.join(orphan_xml_dir, xml_file))
        orphaned_xml_count += 1

# 4. Rename Labels inside the REMAINING valid XML files
renamed_labels_count = 0
remaining_xml_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.xml')]

for xml_file in remaining_xml_files:
    xml_path = os.path.join(target_folder, xml_file)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        modified = False
        
        for obj in root.findall('object'):
            name_tag = obj.find('name')
            if name_tag is not None and name_tag.text in label_map:
                name_tag.text = label_map[name_tag.text]
                modified = True
                renamed_labels_count += 1
        
        if modified:
            tree.write(xml_path)
            
    except Exception as e:
        print(f" Error updating XML {xml_file}: {e}")

# 5. Final Report
print("-" * 35)
print(f" Process Complete!")
print(f" Valid pairs remaining: {len(remaining_xml_files)}")
print(f" Orphaned images moved: {orphaned_images_count}")
print(f" Orphaned XMLs moved: {orphaned_xml_count}")
print(f" Labels corrected: {renamed_labels_count}")
print(f" Cleanup location: {orphan_root}")