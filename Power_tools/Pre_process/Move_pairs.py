import shutil
from pathlib import Path

def separate_unannotated_images(source_folder, destination_folder):
    # Convert paths to pathlib Path objects for easy cross-platform handling
    source = Path(source_folder)
    dest = Path(destination_folder)
    
    # Create the destination folder if it doesn't exist
    dest.mkdir(parents=True, exist_ok=True)
    
    # Define common image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    moved_count = 0
    
    # Iterate through all files in the source folder
    for file_path in source.iterdir():
        # Check if the file is an image
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            
            # Look for an XML file with the exact same name (just replacing the extension)
            expected_xml_path = file_path.with_suffix('.xml')
            
            # If the XML does NOT exist, move the image to the destination folder
            if not expected_xml_path.exists():
                destination_path = dest / file_path.name
                
                # Move the file (Change shutil.move to shutil.copy if you want to keep the originals)
                shutil.move(str(file_path), str(destination_path))
                
                print(f"Moved: {file_path.name}")
                moved_count += 1
                
    print(f"\n Done! Successfully moved {moved_count} images without XMLs.")

if __name__ == "__main__":
    # --- EDIT THESE PATHS ---
    # You can use relative paths (like "my_dataset") or absolute paths (like "C:/images")
    SOURCE_DIR = r"D:\Data_factory\Project_Door\Internet_collection"
    DEST_DIR = r"D:\Data_factory\Project_Door\Internet_collection\pairs"
    
    separate_unannotated_images(SOURCE_DIR, DEST_DIR)