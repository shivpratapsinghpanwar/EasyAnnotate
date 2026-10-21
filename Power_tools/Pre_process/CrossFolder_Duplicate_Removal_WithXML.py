import argparse
import os
import shutil
import hashlib
from pathlib import Path

# --- Configuration ---
parser = argparse.ArgumentParser(description="Move exact-duplicate images (and their XMLs) found across folders.")
parser.add_argument("source_folders", nargs="+", help="folders to scan, in order; the first copy seen is kept")
parser.add_argument("--dump", required=True, help="folder the duplicates are moved to")
args = parser.parse_args()

# Add all your folder paths here
source_folders = args.source_folders

# Where to put the duplicates
duplicate_dump_path = args.dump

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

os.makedirs(duplicate_dump_path, exist_ok=True)

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Global record: {hash_value: original_file_path}
seen_hashes = {}
moved_count = 0

print(f" Starting Cross-Folder De-duplication...")

for folder in source_folders:
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f" Folder not found: {folder}")
        continue
    
    print(f"Scanning: {folder_path.name}...")
    
    # List all files in current folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(image_extensions):
            img_path = folder_path / file_name
            
            # Calculate fingerprint
            img_hash = get_file_hash(img_path)
            
            if img_hash in seen_hashes:
                # DUPLICATE FOUND
                moved_count += 1
                base_name = img_path.stem
                xml_name = f"{base_name}.xml"
                src_xml = folder_path / xml_name
                
                # Create a unique name for the dump folder to avoid overwrites
                # Format: FolderName_FileName
                unique_name_prefix = f"{folder_path.name}_{file_name}"
                unique_xml_prefix = f"{folder_path.name}_{xml_name}"
                
                # Move Image
                shutil.move(str(img_path), os.path.join(duplicate_dump_path, unique_name_prefix))
                
                # Move XML (if it exists)
                if src_xml.exists():
                    shutil.move(str(src_xml), os.path.join(duplicate_dump_path, unique_xml_prefix))
                
                print(f" Moved duplicate: {file_name} from {folder_path.name} (Match found in {seen_hashes[img_hash]})")
            else:
                # NEW UNIQUE IMAGE
                seen_hashes[img_hash] = f"{folder_path.name}/{file_name}"

print("-" * 30)
print(f" Finished!")
print(f"Total duplicates moved: {moved_count}")
print(f"Location: {duplicate_dump_path}")