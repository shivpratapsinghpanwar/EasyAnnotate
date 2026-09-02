import os

# ========= CONFIG =========
folder_path = r"D:\Data_factory\Project_Door\Internet_Images\pairs"
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
xml_extension = ".xml"
# ==========================

# Store base names
image_files = {}
xml_files = {}


# Scan folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    if not os.path.isfile(file_path):
        continue

    base_name, ext = os.path.splitext(file_name)
    ext = ext.lower()

    if ext in image_extensions:
        image_files[base_name] = file_name

    elif ext == xml_extension:
        xml_files[base_name] = file_name

# Find unmatched images
for base_name, image_name in image_files.items():
    if base_name not in xml_files:
        image_path = os.path.join(folder_path, image_name)

        os.remove(image_path)
        print(f"Deleted image without XML: {image_name}")

# Find unmatched XMLs
for base_name, xml_name in xml_files.items():
    if base_name not in image_files:
        xml_path = os.path.join(folder_path, xml_name)

        os.remove(xml_path)
        print(f"Deleted XML without image: {xml_name}")

print("\nCleanup complete.")