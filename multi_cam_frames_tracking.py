import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.reid.utils import FeatureExtractor
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
import random

# ---------------- CONFIG ----------------
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "EPFL-RLC_dataset", "frames")
CAM_FOLDERS = ["cam0", "cam1", "cam2"]

# Detection and tracking parameters
CONFIDENCE_THRESHOLD = 0.45     # Higher = fewer false positives
MATCH_THRESHOLD = 0.65          # Lower = more strict matching (0.5-0.8 recommended)
MAX_FEATURES_PER_ID = 20        # Number of feature vectors to keep per person
IOU_THRESHOLD = 0.3             # For tracking consistency within same camera
FEATURE_UPDATE_INTERVAL = 3     # Update features every N frames
SPATIAL_DISTANCE_THRESHOLD = 200  # Min pixel distance for same ID in same camera

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

yolo_model = YOLO("yolov8n.pt")
reid_model = FeatureExtractor(model_name='osnet_x1_0', device=device)

# ---------------- GLOBAL STORAGE ----------------
# Store feature vectors for each global ID
global_id_features = defaultdict(lambda: deque(maxlen=MAX_FEATURES_PER_ID))
# Assign unique colors to each person for visualization
id_colors = {}
# Track last known position of each ID per camera for temporal consistency
last_positions = defaultdict(lambda: defaultdict(lambda: None))  # {gid: {cam_id: (x, y)}}
id_frame_history = defaultdict(int)  # Track when each ID was last seen
next_gid = 1
frame_count = 0

# ---------------- UTILS ----------------
def get_random_color(seed):
    """Generate consistent random color for each ID"""
    random.seed(seed)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def normalize_feature(feat):
    """Normalize feature vector to unit length"""
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat

def extract_reid_feature(frame, box):
    """Extract ReID feature from detected person crop"""
    x1, y1, x2, y2 = map(int, box)
    
    # Add some padding and ensure valid crop
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    
    # Resize to standard ReID input size
    crop = cv2.resize(crop, (128, 256))
    
    # Extract feature
    try:
        feat = reid_model([crop])[0].cpu().numpy()
        return normalize_feature(feat)
    except:
        return None

def compute_feature_similarity(feat1, feat2):
    """Compute cosine similarity between two features (higher is more similar)"""
    return 1 - cosine(feat1, feat2)

def match_to_global_id(feature, cam_id, position, frame_idx, used_ids_in_frame):
    """Match a feature to existing global IDs or create new one with spatial consistency"""
    global next_gid
    
    if feature is None:
        return None
    
    candidates = []  # List of (gid, similarity, distance_penalty)
    
    # Compare with all existing global IDs
    for gid, feature_list in global_id_features.items():
        if len(feature_list) == 0:
            continue
        
        # Skip if this ID is already used in this frame for this camera
        if gid in used_ids_in_frame.get(cam_id, set()):
            continue
        
        # Compute average similarity with all stored features for this ID
        similarities = [compute_feature_similarity(feature, f) for f in feature_list]
        avg_similarity = np.mean(similarities)
        
        # Apply temporal consistency bonus if this ID was recently seen in this camera
        temporal_bonus = 0
        if gid in last_positions and cam_id in last_positions[gid]:
            last_pos = last_positions[gid][cam_id]
            if last_pos is not None:
                # Calculate distance from last known position
                dist = np.sqrt((position[0] - last_pos[0])**2 + (position[1] - last_pos[1])**2)
                # If close to last position, give bonus (person likely hasn't moved much)
                if dist < SPATIAL_DISTANCE_THRESHOLD:
                    temporal_bonus = 0.05
        
        total_similarity = avg_similarity + temporal_bonus
        candidates.append((gid, total_similarity))
    
    # Find best match
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_gid, best_similarity = candidates[0]
        
        # Match threshold: if similarity is high enough, assign to existing ID
        if best_similarity > MATCH_THRESHOLD:
            # Add this feature to the existing ID's feature bank
            global_id_features[best_gid].append(feature)
            # Update last position
            last_positions[best_gid][cam_id] = position
            id_frame_history[best_gid] = frame_idx
            return best_gid
    
    # Create new global ID
    new_gid = next_gid
    global_id_features[new_gid].append(feature)
    id_colors[new_gid] = get_random_color(new_gid)
    last_positions[new_gid][cam_id] = position
    id_frame_history[new_gid] = frame_idx
    next_gid += 1
    return new_gid

# ---------------- LOAD FRAMES ----------------
def load_camera_frames():
    cams = []
    for cam in CAM_FOLDERS:
        path = os.path.join(BASE_PATH, cam)
        frames = sorted(os.listdir(path))
        frames = [os.path.join(path, f) for f in frames if f.endswith((".jpg", ".jpeg", ".png"))]
        cams.append(frames)
    return cams

# ---------------- MAIN ----------------
def main():
    global frame_count
    
    print("Loading camera frames...")
    cams = load_camera_frames()
    min_len = min(len(cam) for cam in cams)
    
    print(f"Found {len(cams)} cameras with {min_len} frames each")
    print(f"Detection confidence: {CONFIDENCE_THRESHOLD}, Match threshold: {MATCH_THRESHOLD}")
    print("Processing frames... Press 'q' to quit\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    for frame_idx in range(min_len):
        frame_count = frame_idx
        print(f"\rProcessing frame {frame_idx+1}/{min_len} | Active IDs: {len(global_id_features)}", 
              end="", flush=True)
        
        # Load frames from all cameras
        frames = [cv2.imread(cam[frame_idx]) for cam in cams]
        
        # Store all detections across cameras for this frame
        all_detections = []
        # Track which IDs are already used in each camera in this frame
        used_ids_per_camera = defaultdict(set)
        
        # Process each camera
        for cam_id, frame in enumerate(frames):
            if frame is None:
                continue
                
            # Run YOLO detection
            results = yolo_model.track(
                frame,
                persist=True,
                classes=[0],  # Person class
                conf=CONFIDENCE_THRESHOLD,
                imgsz=640,
                verbose=False,
                tracker="bytetrack.yaml"  # Better tracker
            )
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                # Collect all features for this camera first
                cam_detections = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Extract ReID feature for cross-camera matching
                    feature = extract_reid_feature(frame, box)
                    
                    if feature is not None:
                        cam_detections.append({
                            'box': box,
                            'center': center,
                            'feature': feature
                        })
                
                # Now match each detection to global IDs
                # Process in order of detection confidence (larger boxes first)
                for det in cam_detections:
                    # Match to global ID with spatial consistency
                    gid = match_to_global_id(
                        det['feature'], 
                        cam_id, 
                        det['center'], 
                        frame_idx,
                        used_ids_per_camera
                    )
                    
                    if gid is not None:
                        # Mark this ID as used in this camera for this frame
                        used_ids_per_camera[cam_id].add(gid)
                        
                        all_detections.append({
                            'cam_id': cam_id,
                            'box': det['box'],
                            'gid': gid,
                            'feature': det['feature']
                        })
        
        # Visualize all cameras with detections
        processed = visualize_detections(frames, all_detections, frame_idx, min_len)
        
        # Create grid layout
        grid = create_grid_layout(processed)
        
        # Initialize video writer
        if out is None:
            h, w = grid.shape[:2]
            out = cv2.VideoWriter("output_frames.mp4", fourcc, 20, (w, h))
        
        out.write(grid)
        
        # Display
        cv2.imshow("Multi-Camera Person Tracking", grid)
        
        # Wait time: 1ms for fast processing, or 30ms for visible playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n\nStopped by user")
            break
    
    print("\n\nProcessing complete!")
    print(f"Total unique persons tracked: {len(global_id_features)}")
    print(f"Output saved to: output_frames.mp4")
    print("\nPress any key in the video window to close...")
    
    if out:
        out.release()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_detections(frames, detections, frame_idx, total_frames):
    """Draw bounding boxes and labels on all camera frames"""
    processed = []
    
    for cam_id, frame in enumerate(frames):
        if frame is None:
            continue
            
        display = frame.copy()
        
        # Get detections for this camera
        cam_detections = [d for d in detections if d['cam_id'] == cam_id]
        
        # Draw each detection
        for det in cam_detections:
            box = det['box']
            gid = det['gid']
            
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for this ID
            color = id_colors.get(gid, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label = f"ID-{gid}"
            font_scale = 0.8
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Background rectangle for label
            cv2.rectangle(display, 
                         (x1, y1 - label_h - 10), 
                         (x1 + label_w + 10, y1),
                         color, -1)
            
            # Text
            cv2.putText(display, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Camera label
        cv2.rectangle(display, (0, 0), (250, 50), (0, 0, 0), -1)
        cv2.putText(display, f"CAMERA {cam_id}", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Frame counter
        frame_text = f"Frame {frame_idx+1}/{total_frames}"
        cv2.rectangle(display, (0, display.shape[0]-40), (300, display.shape[0]), (0, 0, 0), -1)
        cv2.putText(display, frame_text, (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection count
        count_text = f"Persons: {len(cam_detections)}"
        cv2.putText(display, count_text, (10, display.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        processed.append(display)
    
    return processed

def create_grid_layout(frames):
    """Create a compact grid layout for multiple cameras"""
    if len(frames) == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    if len(frames) == 1:
        return frames[0]
    
    # Target dimensions for each camera view
    target_width = 640   # Width per camera
    target_height = 360  # Height per camera
    
    # Resize all frames to same size
    resized = []
    for f in frames:
        resized_frame = cv2.resize(f, (target_width, target_height))
        resized.append(resized_frame)
    
    if len(frames) == 2:
        # Side by side for 2 cameras
        row = np.hstack(resized)
        return row
    
    elif len(frames) == 3:
        # For 3 cameras: 2 on top, 1 on bottom (centered)
        # Top row: cameras 0 and 1
        top_row = np.hstack([resized[0], resized[1]])
        
        # Bottom row: camera 2 centered
        # Create padding to center the third camera
        padding_width = target_width // 2
        left_padding = np.zeros((target_height, padding_width, 3), dtype=np.uint8)
        right_padding = np.zeros((target_height, padding_width, 3), dtype=np.uint8)
        bottom_row = np.hstack([left_padding, resized[2], right_padding])
        
        # Stack vertically
        grid = np.vstack([top_row, bottom_row])
        return grid
    
    elif len(frames) == 4:
        # 2x2 grid for 4 cameras
        top_row = np.hstack(resized[:2])
        bottom_row = np.hstack(resized[2:4])
        grid = np.vstack([top_row, bottom_row])
        return grid
    
    else:
        # For more cameras, create rows of 3
        rows = []
        for i in range(0, len(frames), 3):
            row_frames = resized[i:i+3]
            # Pad if necessary
            while len(row_frames) < 3:
                row_frames.append(np.zeros((target_height, target_width, 3), dtype=np.uint8))
            row = np.hstack(row_frames)
            rows.append(row)
        return np.vstack(rows)

if __name__ == "__main__":
    main()