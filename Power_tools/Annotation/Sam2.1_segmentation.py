import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing
import queue

# --- CONFIGURATION ---
IMAGE_DIR = r"D:\Data_factory\Project_Door\Internet_collection\duplicatess"
OUTPUT_DIR = r"D:\Data_factory\Project_Door\Internet_collection\duplicatess\Mask"
MODEL_CFG = r"D:\Data_factory\Models\Sam2.1_tiny\sam2.1_hiera_t.yaml"
MODEL_WEIGHTS = r"D:\Data_factory\Models\Sam2.1_tiny\sam2.1_hiera_tiny.pt"
DEVICE = "cpu"


DISPLAY_HEIGHT = 800  
UI_BAR_HEIGHT = 80    
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- THE SAM 2 WORKER PROCESS ---
# This runs completely isolated from the UI, bypassing the GIL.
def sam_worker(input_queue, output_queue):
    print("[Worker] Initializing PyTorch and SAM 2...")
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Initialize model once
    sam2_model = build_sam2(MODEL_CFG, MODEL_WEIGHTS, device=DEVICE)
    predictor_active = SAM2ImagePredictor(sam2_model)
    predictor_bg = SAM2ImagePredictor(sam2_model)
    print("[Worker] Ready.")

    while True:
        try:
            command, payload = input_queue.get()
            
            if command == "QUIT":
                break
                
            elif command == "INITIAL_LOAD":
                img_path = payload
                img = cv2.imread(img_path)
                predictor_active.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                output_queue.put(("INITIAL_LOAD_DONE", None))
                
            elif command == "PRELOAD_NEXT":
                img_path = payload
                img = cv2.imread(img_path)
                predictor_bg.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                output_queue.put(("PRELOAD_DONE", None))
                
            elif command == "SWAP":
                predictor_active, predictor_bg = predictor_bg, predictor_active
                output_queue.put(("SWAP_DONE", None))
                
            elif command == "PREDICT":
                box = payload
                masks, _, _ = predictor_active.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=False 
                )
                mask_binary = (masks[0] > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    output_queue.put(("PREDICT_DONE", largest_contour))
                else:
                    output_queue.put(("PREDICT_DONE", None))
                    
        except Exception as e:
            print(f"[Worker] Error: {e}")

# --- MAIN UI PROCESS STATE ---
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))])
current_idx = 0

current_image_full = None
display_image_clean = None
display_canvas = None
saved_objects = []
scale_factor = 1.0
box_start = None

# Multiprocessing Pipes
input_q = None
output_q = None
bg_is_encoding = False

# --- UI DRAWING ---
def draw_ui(canvas, status_text=""):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], UI_BAR_HEIGHT), (30, 30, 30), -1)
    
    buttons = [
        (20, 20, 120, 60, "PREV", (0, 100, 200)),
        (140, 20, 240, 60, "UNDO", (0, 140, 255)),
        (260, 20, 360, 60, "NEXT", (0, 200, 0)),
        (380, 20, 480, 60, "QUIT", (0, 0, 200))
    ]
    
    for (x1, y1, x2, y2, text, color) in buttons:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.putText(canvas, text, (x1 + 15, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    info = f"Img: {current_idx + 1}/{len(image_files)} | Rocks: {len(saved_objects)} | {status_text}"
    cv2.putText(canvas, info, (520, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# --- XML HANDLING ---
def save_xml():
    if not saved_objects: return
    filename = image_files[current_idx]
    h, w = current_image_full.shape[:2]
    
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    
    for obj in saved_objects:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = "rock"
        bbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(obj['bbox'][0])
        ET.SubElement(bbox, "ymin").text = str(obj['bbox'][1])
        ET.SubElement(bbox, "xmax").text = str(obj['bbox'][2])
        ET.SubElement(bbox, "ymax").text = str(obj['bbox'][3])
        poly = ET.SubElement(obj_elem, "polygon")
        for pt in obj['polygon']:
            pt_elem = ET.SubElement(poly, "pt")
            ET.SubElement(pt_elem, "x").text = str(pt[0][0])
            ET.SubElement(pt_elem, "y").text = str(pt[0][1])

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    xml_path = os.path.join(OUTPUT_DIR, filename.rsplit('.', 1)[0] + '.xml')
    with open(xml_path, "w") as f: f.write(xml_str)

def delete_current_xml():
    xml_path = os.path.join(OUTPUT_DIR, image_files[current_idx].rsplit('.', 1)[0] + '.xml')
    if os.path.exists(xml_path): os.remove(xml_path)

# --- CORE PIPELINE ---
def render_base_image(status="Ready"):
    global display_canvas, display_image_clean
    display_image_clean = cv2.resize(current_image_full, (int(current_image_full.shape[1] * scale_factor), DISPLAY_HEIGHT))
    
    for obj in saved_objects:
        # CORRECTED MATH: Unpack the actual minimum and maximum coordinates
        xmin, ymin, xmax, ymax = obj["bbox"]
        
        # Scale the box corners directly to the display size
        dx1, dy1 = int(xmin * scale_factor), int(ymin * scale_factor)
        dx2, dy2 = int(xmax * scale_factor), int(ymax * scale_factor)
        
        # Draw perfectly aligned bounding box
        cv2.rectangle(display_image_clean, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
        
        # Draw the perfectly aligned polygon mask
        scaled_poly = (obj["polygon"] * scale_factor).astype(np.int32)
        cv2.drawContours(display_image_clean, [scaled_poly], -1, (0, 255, 0), 2)
        
    display_canvas[UI_BAR_HEIGHT:, :] = display_image_clean
    draw_ui(display_canvas, status_text=status)
    cv2.imshow("Annotator", display_canvas)

# --- MOUSE LOGIC ---
def mouse_callback(event, x, y, flags, param):
    global display_canvas, saved_objects, box_start, scale_factor
    
    if event == cv2.EVENT_LBUTTONDOWN and y < UI_BAR_HEIGHT:
        if 20 <= x <= 120:    navigate(-1)
        elif 140 <= x <= 240: 
            if box_start: box_start = None; render_base_image("Box cancelled.")
            elif saved_objects: saved_objects.pop(); render_base_image("Removed last rock.")
        elif 260 <= x <= 360: navigate(1)
        elif 380 <= x <= 480: shutdown_and_exit()
        return
        
    img_y = max(0, y - UI_BAR_HEIGHT)
    orig_x, orig_y = int(x / scale_factor), int(img_y / scale_factor)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if box_start is None:
            box_start = (orig_x, orig_y)
            render_base_image(status="Click opposite corner to lock rock.")
        else:
            xmin, ymin = min(box_start[0], orig_x), min(box_start[1], orig_y)
            xmax, ymax = max(box_start[0], orig_x), max(box_start[1], orig_y)
            box_start = None
            
            if xmax - xmin < 10 or ymax - ymin < 10:
                render_base_image(status="Box too small, cancelled.")
                return
            
            # Send prediction command to Worker Process
            render_base_image(status="Predicting mask...")
            cv2.waitKey(1)
            input_q.put(("PREDICT", np.array([xmin, ymin, xmax, ymax])))
            
            # Block UI slightly to wait for the small mask return
            cmd, largest_contour = output_q.get() 
            if cmd == "PREDICT_DONE" and largest_contour is not None:
                bx, by, bw, bh = cv2.boundingRect(largest_contour)
                saved_objects.append({
                    "bbox": (bx, by, bx+bw, by+bh),
                    "polygon": largest_contour
                })
            render_base_image(status="Rock Saved! Next rock.")
            
    elif event == cv2.EVENT_RBUTTONDOWN and box_start:
        box_start = None
        render_base_image("Box drawing cancelled.")
        
    elif event == cv2.EVENT_MOUSEMOVE and box_start:
        temp_canvas = display_image_clean.copy()
        disp_sx, disp_sy = int(box_start[0]*scale_factor), int(box_start[1]*scale_factor)
        cv2.rectangle(temp_canvas, (disp_sx, disp_sy), (x, img_y), (0, 255, 255), 2)
        display_canvas[UI_BAR_HEIGHT:, :] = temp_canvas
        cv2.imshow("Annotator", display_canvas)

# --- NAVIGATION & PROCESS PIPELINE ---
def navigate(direction):
    global current_idx, current_image_full, scale_factor, saved_objects
    global box_start, display_canvas, bg_is_encoding
    
    if direction == 1: 
        save_xml()
        if current_idx < len(image_files) - 1:
            # Wait for Worker to finish preloading if it hasn't yet
            if bg_is_encoding:
                render_base_image(status="Waiting for background encoder...")
                cv2.waitKey(1)
                cmd, _ = output_q.get() 
                bg_is_encoding = False
                
            # Tell Worker to swap its internal predictors
            input_q.put(("SWAP", None))
            output_q.get() 
            
            current_idx += 1
            load_ui_image_state(image_files[current_idx])
            
            # Trigger next preload asynchronously
            if current_idx < len(image_files) - 1:
                next_path = os.path.join(IMAGE_DIR, image_files[current_idx + 1])
                input_q.put(("PRELOAD_NEXT", next_path))
                bg_is_encoding = True
                
            render_base_image(status="Ready.")
        else:
            print("Dataset Complete!")
            shutdown_and_exit()
            
    elif direction == -1: 
        if current_idx > 0:
            delete_current_xml()
            current_idx -= 1
            delete_current_xml()
            
            load_ui_image_state(image_files[current_idx])
            render_base_image("Hard loading previous image...")
            cv2.waitKey(1)
            
            input_q.put(("INITIAL_LOAD", os.path.join(IMAGE_DIR, image_files[current_idx])))
            output_q.get() 
            bg_is_encoding = False
            
            if current_idx < len(image_files) - 1:
                next_path = os.path.join(IMAGE_DIR, image_files[current_idx + 1])
                input_q.put(("PRELOAD_NEXT", next_path))
                bg_is_encoding = True
                
            render_base_image("Ready.")

def load_ui_image_state(img_name):
    global current_image_full, scale_factor, display_canvas, saved_objects, box_start
    current_image_full = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    orig_h, orig_w = current_image_full.shape[:2]
    scale_factor = DISPLAY_HEIGHT / float(orig_h)
    display_canvas = np.zeros((DISPLAY_HEIGHT + UI_BAR_HEIGHT, int(orig_w * scale_factor), 3), dtype=np.uint8)
    saved_objects = []
    box_start = None

def shutdown_and_exit():
    input_q.put(("QUIT", None))
    cv2.destroyAllWindows()
    exit()

# --- ENTRY POINT ---
if __name__ == '__main__':
    # Required for Windows multiprocessing safety
    multiprocessing.freeze_support()
    
    if len(image_files) == 0:
        print("No images found in directory.")
        exit()

    # Create IPC Queues
    input_q = multiprocessing.Queue()
    output_q = multiprocessing.Queue()
    
    # Spawn the isolated PyTorch Process
    worker = multiprocessing.Process(target=sam_worker, args=(input_q, output_q))
    worker.start()
    
    # Initial Setup
    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_callback)
    
    load_ui_image_state(image_files[0])
    render_base_image(status="Worker Process loading AI... Please wait.")
    cv2.waitKey(1)
    
    # Force the first image load
    input_q.put(("INITIAL_LOAD", os.path.join(IMAGE_DIR, image_files[0])))
    output_q.get() 
    
    # Start pre-loading the second image immediately
    if len(image_files) > 1:
        next_path = os.path.join(IMAGE_DIR, image_files[1])
        input_q.put(("PRELOAD_NEXT", next_path))
        bg_is_encoding = True

    render_base_image(status="Ready. Click top-left of a rock.")

    # Main UI Loop
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('n'): navigate(1)
        elif key == ord('p'): navigate(-1)
        elif key == ord('u'): 
            if box_start: box_start = None; render_base_image("Box cancelled.")
            elif saved_objects: saved_objects.pop(); render_base_image("Removed last rock.")
        elif key == ord('q') or key == 27:
            shutdown_and_exit()
            
        # Non-blocking check to see if background worker finished early
        if bg_is_encoding:
            try:
                msg, _ = output_q.get_nowait()
                if msg == "PRELOAD_DONE":
                    bg_is_encoding = False
            except queue.Empty:
                pass