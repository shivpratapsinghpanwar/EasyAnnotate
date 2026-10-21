import argparse
import os
import shutil
from PIL import Image
import imagehash

# --- Configuration ---
parser = argparse.ArgumentParser(description="Move visually similar images into a duplicates subfolder.")
parser.add_argument("master_folder", help="folder of images to de-duplicate")
master_folder = parser.parse_args().master_folder
duplicates_folder = os.path.join(master_folder, "duplicatess_10")

# Create the duplicates folder if it doesn't exist
os.makedirs(duplicates_folder, exist_ok=True)

# Tolerance for similarity (Hamming Distance)
# 0 = Exact visual match
# 1-10 = Very similar (catches minor crops, compressions, color shifts)
# >10 = Might start catching images that are fundamentally different
SIMILARITY_THRESHOLD = 10

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

print(f"Scanning for visually similar images (Threshold: {SIMILARITY_THRESHOLD})...")

# Dictionary to store kept images -> {filepath: phash}
kept_images = {}
duplicate_count = 0

for file_name in os.listdir(master_folder):
    if file_name.lower().endswith(image_extensions):
        file_path = os.path.join(master_folder, file_name)
        
        # Skip if it's a directory
        if os.path.isdir(file_path):
            continue
            
        try:
            # Open image and calculate perceptual hash
            img = Image.open(file_path)
            current_hash = imagehash.phash(img)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        is_duplicate = False
        
        # Compare current image against all previously kept images
        for kept_path, kept_hash in kept_images.items():
            # Subtracting two hashes gives the Hamming distance (difference)
            hash_difference = current_hash - kept_hash
            
            if hash_difference <= SIMILARITY_THRESHOLD:
                is_duplicate = True
                break # No need to check further, we found a match

        if is_duplicate:
            # Paths for the image
            dst_image = os.path.join(duplicates_folder, file_name)
            
            # Paths for the corresponding XML (Uncomment to use)
            # xml_name = os.path.splitext(file_name)[0] + ".xml"
            # src_xml = os.path.join(master_folder, xml_name)
            # dst_xml = os.path.join(duplicates_folder, xml_name)
            
            # Move the Image
            shutil.move(file_path, dst_image)
            
            # Move the XML if it exists (Uncomment to use)
            # if os.path.exists(src_xml):
            #     shutil.move(src_xml, dst_xml)
            
            duplicate_count += 1
            print(f"Moved duplicate: {file_name}")
        else:
            # If it's unique, add it to our kept ledger
            kept_images[file_path] = current_hash

print("-" * 30)
print(f"Clean-up complete!")
print(f"Total similar pairs moved to subfolder: {duplicate_count}")