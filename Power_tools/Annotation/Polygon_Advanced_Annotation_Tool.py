"""
Universal Polygon Image Labeling Tool (With Background Logic)
=============================================================
Features:
    - Custom labels loaded dynamically from text files or default fallbacks
    - Polygon Annotation (N-sided shapes, 3 to 100+ points)
    - Hybrid XML export (saves both <bndbox> and <polygon> coordinates)
    - Background Image Logic (Negative sampling for model training)

Usage:
    python UniversalAnnotator.py --image <path_to_image> --labels <path_to_labels.txt>
    python UniversalAnnotator.py --folder <path_to_folder> --labels <path_to_labels.txt>

Controls:
    - Left Click         : Trace points around the object (draws lines)
    - Right Click        : Close the polygon shape
    - C                  : Cancel current temporary points
    - U / Ctrl+Z         : Undo last completed shape
    - B                  : Mark as Background (Saves empty XML & goes to next image)
    - S                  : Save XML annotation file manually
    - N                  : Next image (folder mode)
    - P                  : Previous image (folder mode)
    - Q / ESC            : Quit
    - 1-9                : Switch active label class
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import argparse
import glob
from pathlib import Path
from PIL import Image

# Global placeholders populated dynamically at runtime
LABELS = []
LABEL_COLORS = []

# ─────────────────────────────────────────────────────────────────────────────
#  Image conversion to JPG
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_jpg(image_path):
    _, ext = os.path.splitext(image_path)
    if ext.lower() in ('.jpg', '.jpeg'):
        return image_path
    
    try:
        img = Image.open(image_path)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        jpg_path = os.path.splitext(image_path)[0] + '.jpg'
        img.save(jpg_path, 'JPEG', quality=95)
        print(f"[CONVERT] {os.path.basename(image_path)} → {os.path.basename(jpg_path)}")
        return jpg_path
    except Exception as e:
        print(f"[ERROR] Failed to convert {image_path}: {e}")
        return image_path

# ─────────────────────────────────────────────────────────────────────────────
#  XML Generation 
# ─────────────────────────────────────────────────────────────────────────────

def build_xml(image_path, width, height, boxes):
    filename = os.path.basename(image_path)
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text     = filename

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text  = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text  = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    # If 'boxes' is empty (Background Image), this loop simply won't run, 
    # resulting in a perfectly valid XML with zero <object> tags.
    for (points, label) in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text      = label
        ET.SubElement(obj, "pose").text      = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "occluded").text  = "0"

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(min(x_coords))
        ET.SubElement(bndbox, "xmax").text = str(max(x_coords))
        ET.SubElement(bndbox, "ymin").text = str(min(y_coords))
        ET.SubElement(bndbox, "ymax").text = str(max(y_coords))
        
        polygon = ET.SubElement(obj, "polygon")
        for i, (px, py) in enumerate(points, 1):
            ET.SubElement(polygon, f"x{i}").text = str(int(px))
            ET.SubElement(polygon, f"y{i}").text = str(int(py))

    ET.SubElement(annotation, "metadata").text = " "

    raw  = ET.tostring(annotation, encoding="unicode")
    dom  = minidom.parseString(raw)
    return dom.toprettyxml(indent="  ")

def save_xml(image_path, width, height, boxes):
    xml_str  = build_xml(image_path, width, height, boxes)
    base     = os.path.splitext(image_path)[0]
    xml_path = base + ".xml"
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return xml_path

# ─────────────────────────────────────────────────────────────────────────────
#  UI & Display Renderers
# ─────────────────────────────────────────────────────────────────────────────

def draw_boxes(canvas, boxes, current_points, current_mouse, current_label_idx, y_offset=0):
    for (points, label) in boxes:
        disp_points = [(px, py + y_offset) for (px, py) in points]
        pts_array = np.array(disp_points, np.int32).reshape((-1, 1, 2))
        
        idx   = LABELS.index(label) if label in LABELS else 0
        color = LABEL_COLORS[idx]
        
        cv2.polylines(canvas, [pts_array], isClosed=True, color=color, thickness=2)
        
        top_y = min(p[1] for p in disp_points)
        left_x = min(p[0] for p in disp_points)
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        badge_y1 = max(top_y - th - 8, 0)
        cv2.rectangle(canvas, (left_x, badge_y1), (left_x + tw + 8, top_y), color, -1)
        cv2.putText(canvas, label, (left_x + 4, top_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    if current_points:
        color = LABEL_COLORS[current_label_idx]
        for px, py in current_points:
            cv2.circle(canvas, (px, py), 4, color, -1)   
            cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1) 
            
        if len(current_points) > 1:
            pts_array = np.array(current_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_array], isClosed=False, color=color, thickness=2)
            
        if current_mouse:
            last_pt = current_points[-1]
            cv2.line(canvas, last_pt, current_mouse, color, 2)

def draw_ui(canvas, label_idx, boxes, status_msg, img_idx=None, total=None):
    h, w = canvas.shape[:2]

    # Top HUD
    cv2.rectangle(canvas, (0, 0), (w, 38), (20, 20, 20), -1)
    color = LABEL_COLORS[label_idx]
    label = LABELS[label_idx]
    pill  = f"  [{label_idx+1}] {label}  "
    (pw, ph), _ = cv2.getTextSize(pill, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(canvas, (6, 5), (6 + pw + 4, 33), color, -1)
    cv2.putText(canvas, pill, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    info = f"Shapes: {len(boxes)}"
    if img_idx is not None:
        info += f"   Image {img_idx+1}/{total}"
    cv2.putText(canvas, info, (pw + 30, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Bottom HUD
    cv2.rectangle(canvas, (0, h - 34), (w, h), (20, 20, 20), -1)
    cv2.putText(canvas, status_msg, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    btn_x, btn_y = w - 110, h - 34
    cv2.rectangle(canvas, (btn_x, btn_y), (w - 6, h - 6), (60, 60, 200), -1)
    cv2.rectangle(canvas, (btn_x, btn_y), (w - 6, h - 6), (120, 120, 255), 1)
    cv2.putText(canvas, "UNDO [U]", (btn_x + 6, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    hint = "L-Click=Point  R-Click=Close  B=Background  C=Cancel  S=Save  N=Next  1-9=Label"
    cv2.putText(canvas, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 130, 130), 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  Mouse State Engine
# ─────────────────────────────────────────────────────────────────────────────

class DrawState:
    def __init__(self):
        self.points     = []     
        self.current    = None   
        self.done       = False  

    def reset(self):
        self.points     = []
        self.current    = None
        self.done       = False

def make_mouse_callback(state):
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            state.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(state.points) >= 3:
                state.done = True
            else:
                state.reset() 
    return callback

# ─────────────────────────────────────────────────────────────────────────────
#  Label Execution Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def label_image(image_path, img_idx=None, total=None):
    jpg_path = convert_to_jpg(image_path)
    orig = cv2.imread(jpg_path)
    if orig is None:
        print(f"[WARN] Cannot open target file: {jpg_path}")
        return True

    h, w  = orig.shape[:2]
    HUD_TOP, HUD_BOTTOM = 40, 40
    canvas_h = h + HUD_TOP + HUD_BOTTOM

    boxes = []  
    label_idx = 0
    status    = "Trace the object. Right-Click to close. Press 'B' for Background Image."

    state      = DrawState()
    state.done = False

    win_name = "Universal Polygon Labeler"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    win_w, win_h = max(w, 800), max(canvas_h, 600)
    cv2.resizeWindow(win_name, win_w, win_h)
    cv2.setMouseCallback(win_name, make_mouse_callback(state))

    while True:
        canvas = np.zeros((canvas_h, w, 3), dtype=np.uint8)
        canvas[HUD_TOP : HUD_TOP + h, 0:w] = orig

        draw_boxes(canvas, boxes, state.points, state.current, label_idx, y_offset=HUD_TOP)
        draw_ui(canvas, label_idx, boxes, status, img_idx, total)   

        if state.done:
            img_points = [(px, py - HUD_TOP) for px, py in state.points]
            clamped_points = []
            for px, py in img_points:
                cx = max(0, min(px, w - 1))
                cy = max(0, min(py, h - 1))
                clamped_points.append((cx, cy))
            
            boxes.append((clamped_points, LABELS[label_idx]))
            status = f"Polygon saved: {LABELS[label_idx]} ({len(clamped_points)} points)"
            
            state.reset()
            state.done = False

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):          
            cv2.destroyAllWindows()
            return False
            
        elif key == ord('c'):              
            state.reset()
            status = "Selection wiped clean."
            
        elif key in (ord('u'), 26):        
            if boxes:
                removed = boxes.pop()
                status  = f"Undo: Dropped {removed[1]}"
                
        elif key == ord('b'):
            # BACKGROUND LOGIC: Save an empty XML and advance
            save_xml(image_path, w, h, [])
            print(f"[BACKGROUND] Saved empty XML for {os.path.basename(image_path)}")
            cv2.destroyAllWindows()
            return True
            
        elif key == ord('s'):              
            if not boxes:
                status = "Workspace empty. Press 'B' if this is a background image!"
            else:
                path   = save_xml(image_path, w, h, boxes)
                status = f"Saved file → {os.path.basename(path)}"
                
        elif key == ord('n'):              
            if boxes:
                save_xml(image_path, w, h, boxes)
            cv2.destroyAllWindows()
            return True
            
        elif key == ord('p'):              
            cv2.destroyAllWindows()
            return "prev"
            
        elif ord('1') <= key <= ord('9'):
            pressed_idx = key - ord('1')
            if pressed_idx < len(LABELS):
                label_idx = pressed_idx
                status    = f"Active layer → {LABELS[label_idx]}"

# ─────────────────────────────────────────────────────────────────────────────
#  Application Launcher
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global LABELS, LABEL_COLORS
    
    parser = argparse.ArgumentParser(description="Universal Image Labeling Suite")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to file processing target")
    group.add_argument("--folder", help="Path to batch target directory")
    parser.add_argument("--labels", help="Optional path to classes/labels text file")
    args = parser.parse_args()

    if args.labels and os.path.exists(args.labels):
        with open(args.labels, 'r', encoding='utf-8') as f:
            LABELS = [line.strip() for line in f if line.strip()]
    else:
        LABELS = ["class_1", "class_2"]
        print("[INFO] No external label map provided. Loading basic standard classes.")

    BASE_PALETTE = [
        (0, 200, 255), (0, 255, 120), (255, 80, 80), 
        (220, 60, 220), (0, 235, 235), (240, 240, 0),
        (120, 120, 255), (140, 240, 140), (255, 140, 140)
    ]
    LABEL_COLORS = [BASE_PALETTE[i % len(BASE_PALETTE)] for i in range(len(LABELS))]

    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    images = [args.image] if args.image else []
    
    if args.folder:
        for ext in EXTS:
            images += glob.glob(os.path.join(args.folder, ext))
            images += glob.glob(os.path.join(args.folder, ext.upper()))
        images = sorted(set(images))
        
    if not images:
        print("[ERROR] Targets empty. Execution array contains zero images.")
        return

    print(f"\n{'='*60}\n Execution Pipeline Online | Total dataset array: {len(images)} images\n{'='*60}")
    for i, lbl in enumerate(LABELS, 1):
        print(f"  [{i}] {lbl}")
    print(f"\n[HOTKEY ADDED]: Press 'B' to save an image as BACKGROUND (empty XML)")
    print(f"{'='*60}\n")

    idx = 0
    while 0 <= idx < len(images):
        result = label_image(images[idx], img_idx=idx, total=len(images))
        if result is False:
            break
        idx = max(0, idx - 1) if result == "prev" else idx + 1
    print("Session concluded. Dataset updated.")

if __name__ == "__main__":
    main()