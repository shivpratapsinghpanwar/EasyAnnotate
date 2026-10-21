import argparse
import os
import shutil

# ========= CONFIG =========
parser = argparse.ArgumentParser(description="Rename image/XML pairs to <prefix><number>.")
parser.add_argument("folder_path", help="folder of images and XMLs")
parser.add_argument("--prefix", required=True, help="name prefix, e.g. img -> img1.jpg, img1.xml")
parser.add_argument("--start", type=int, default=1, help="first number to use (default: 1)")
args = parser.parse_args()
folder_path = args.folder_path

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

prefix = args.prefix
start_index = args.start
# ==========================

# Get all XML files
xml_files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(".xml")]

counter = start_index

for xml_file in sorted(xml_files):

    xml_base = os.path.splitext(xml_file)[0]

    # Find matching image
    matching_image = None

    for ext in image_extensions:
        possible_image = xml_base + ext

        if os.path.exists(os.path.join(folder_path, possible_image)):
            matching_image = possible_image
            break

    # Skip if image not found
    if matching_image is None:
        print(f"Skipping (no image found): {xml_file}")
        continue

    # Original paths
    old_xml_path = os.path.join(folder_path, xml_file)
    old_img_path = os.path.join(folder_path, matching_image)

    # New names
    image_ext = os.path.splitext(matching_image)[1]

    new_base = f"{prefix}{counter}"

    new_img_name = new_base + image_ext
    new_xml_name = new_base + ".xml"

    new_img_path = os.path.join(folder_path, new_img_name)
    new_xml_path = os.path.join(folder_path, new_xml_name)

    # Temporary names to avoid overwrite conflicts
    temp_img_path = os.path.join(folder_path, f"temp_img_{counter}{image_ext}")
    temp_xml_path = os.path.join(folder_path, f"temp_xml_{counter}.xml")

    shutil.move(old_img_path, temp_img_path)
    shutil.move(old_xml_path, temp_xml_path)

    shutil.move(temp_img_path, new_img_path)
    shutil.move(temp_xml_path, new_xml_path)

    print(f"Renamed:")
    print(f"  {matching_image} -> {new_img_name}")
    print(f"  {xml_file} -> {new_xml_name}")

    counter += 1

print("\nRenaming complete.")           