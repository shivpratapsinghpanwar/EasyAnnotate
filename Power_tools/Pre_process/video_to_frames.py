import argparse
import cv2
import os

# ========= CONFIG =========
parser = argparse.ArgumentParser(description="Save the frames of a video as JPEGs.")
parser.add_argument("video_path", help="path to your video")
parser.add_argument("output_folder", help="folder to save frames")
parser.add_argument("--every", type=int, default=1, help="save every nth frame (default: 1)")
args = parser.parse_args()

video_path = args.video_path
output_folder = args.output_folder
save_every = args.every
# ==========================

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save every nth frame
    if frame_count % save_every == 0:
        frame_name = f"frame_{saved_count:06d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)

        cv2.imwrite(frame_path, frame)

        print(f"Saved: {frame_path}")
        saved_count += 1

    frame_count += 1

cap.release()

print("\nDone.")
print(f"Total frames processed: {frame_count}")
print(f"Frames saved: {saved_count}")