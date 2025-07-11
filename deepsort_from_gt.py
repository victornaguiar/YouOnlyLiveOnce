import os
import cv2
import torch
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# --- Configuration ---
# Define paths to your data
# NOTE: Please double-check these paths match your Google Drive structure
BASE_PATH = "/content/drive/MyDrive/SOCCER_DATA/deepsort_dataset_test"
SEQUENCE_NAME = "SNMOT-116" # The sequence you want to process

DETECTIONS_PATH = os.path.join(BASE_PATH, "detections", f"{SEQUENCE_NAME}.txt")
IMAGE_SEQUENCE_PATH = os.path.join(BASE_PATH, "sequences", SEQUENCE_NAME)
OUTPUT_VIDEO_PATH = f"/content/{SEQUENCE_NAME}_tracked_output.mp4"

# DeepSort configuration
CFG_FILE = "deep_sort_pytorch/configs/deep_sort.yaml"
# --- CORRECTED LINE BELOW (Absolute Path) ---
CHECKPOINT_FILE = "/content/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7" 

# --- Helper Function to Load Detections ---
def load_detections(det_file):
    """
    Loads detections from a MOT-style text file.
    Returns a dictionary where keys are frame numbers and values are lists of detections.
    """
    detections = {}
    with open(det_file, 'r') as f:
        for line in f:
            parts = [float(p) for p in line.strip().split(',')]
            frame_num = int(parts[0])
            
            if frame_num not in detections:
                detections[frame_num] = []
            
            # Format: [x1, y1, x2, y2, confidence]
            # Ground truth confidence is set to 1.0 since it's a known object
            x1, y1, w, h = parts[2:6]
            detections[frame_num].append([x1, y1, x1 + w, y1 + h, 1.0])
            
    return detections

# --- Main Tracking Logic ---
def main():
    # 1. Initialize DeepSort
    print("Initializing DeepSort...")
    cfg = get_config()
    cfg.merge_from_file(CFG_FILE)
    
    # Check if CUDA is available and use it
    use_cuda = torch.cuda.is_available()
    
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=use_cuda
    )
    print("DeepSort initialized.")

    # 2. Load Detections
    print(f"Loading detections from: {DETECTIONS_PATH}")
    all_detections = load_detections(DETECTIONS_PATH)
    print(f"Found detections for {len(all_detections)} frames.")

    # 3. Setup for Video Writing
    print("Setting up video writer...")
    frame_files = sorted([f for f in os.listdir(IMAGE_SEQUENCE_PATH) if f.endswith('.jpg')])
    if not frame_files:
        print(f"Error: No image files found in {IMAGE_SEQUENCE_PATH}")
        return

    # Get frame dimensions from the first image
    first_frame_path = os.path.join(IMAGE_SEQUENCE_PATH, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Could not read the first frame from {first_frame_path}")
        return
    frame_height, frame_width, _ = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 25.0, (frame_width, frame_height))
    print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")

    # 4. Process Each Frame
    print("Starting tracking process...")
    for i, frame_file in enumerate(frame_files):
        frame_num = i + 1 # MOT format is 1-indexed
        
        frame_path = os.path.join(IMAGE_SEQUENCE_PATH, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue

        # Get detections for the current frame
        if frame_num in all_detections:
            bboxes_xyxy = np.array(all_detections[frame_num])
            confs = bboxes_xyxy[:, 4]
            
            # DeepSort expects bboxes in xywh format
            xywhs = bboxes_xyxy.copy()
            xywhs[:, 2] = xywhs[:, 2] - xywhs[:, 0] # width
            xywhs[:, 3] = xywhs[:, 3] - xywhs[:, 1] # height

            # Update the tracker
            outputs = deepsort.update(xywhs, confs, frame)
            
            # Draw bounding boxes and IDs on the frame
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    bboxes = output[0:4]
                    track_id = output[4]
                    
                    x1, y1, x2, y2 = [int(b) for b in bboxes]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        if frame_num % 100 == 0:
            print(f"Processed frame {frame_num}/{len(frame_files)}")
            
    # 5. Release Resources
    out.release()
    print(f"Tracking complete. Video saved to {OUTPUT_VIDEO_PATH}")

# Run the main function
if __name__ == "__main__":
    # Before running, let's check if the necessary files and directories exist
    if not os.path.exists(DETECTIONS_PATH):
        print(f"ERROR: Detections file not found at {DETECTIONS_PATH}")
    elif not os.path.exists(IMAGE_SEQUENCE_PATH):
        print(f"ERROR: Image sequence directory not found at {IMAGE_SEQUENCE_PATH}")
    elif not os.path.exists(CHECKPOINT_FILE):
        print(f"ERROR: DeepSort checkpoint file not found at {CHECKPOINT_FILE}")
        print("Please ensure you have cloned the 'deep_sort_pytorch' repo and the 'ckpt.t7' file is present.")
    else:
        main()
