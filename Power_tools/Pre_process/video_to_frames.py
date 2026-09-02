import cv2
import os

# ========= CONFIG =========
video_path = "D:\Client_data\Archive 2\WhatsApp Video 2026-05-07 at 16.46.30.mp4"   # path to your video
output_folder = r"D:\Client_data\Archive 2\out_vid"       # folder to save frames
save_every = 1                   # save every nth frame
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