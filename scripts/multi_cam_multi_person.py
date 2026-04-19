"""
MULTI CAMERA FRAMES - MULTI PERSON TRACKING
Tracks different people across multiple physical cameras using ReID
"""

print("Running: MULTI CAM MULTI PERSON")

import cv2
import os
import numpy as np
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ultralytics import YOLO
from collections import defaultdict
from src.reid.reid_manager import ReIDManager
from src.utils.visualization import get_color, draw_bbox, draw_label
from src.input.video_input import get_loader


# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "..", "EPFL-RLC_dataset", "frames")
CAM_FOLDERS = ["cam0", "cam1", "cam2"]

CONFIDENCE_THRESHOLD = 0.45
MATCH_THRESHOLD = 0.65
MAX_FEATURES_PER_ID = 20
SPATIAL_DISTANCE_THRESHOLD = 200

# =========================
# GLOBAL STATE FOR DASHBOARD
# =========================
_reid_manager = None
_yolo_model = None
_loader = None
_all_global_ids = None
_initialized = False


def _initialize():
    """Initialize global state (called once)"""
    global _reid_manager, _yolo_model, _loader, _all_global_ids, _initialized
    
    if _initialized:
        return
    
    print("[Multi-Cam-Multi] Initializing...")
    
    # Check if dataset exists
    if not os.path.exists(BASE_PATH):
        print(f"\nError: Dataset not found at {BASE_PATH}")
        raise FileNotFoundError(f"Dataset not found at {BASE_PATH}")
    
    # Initialize ReID manager
    _reid_manager = ReIDManager(
        model_name='osnet_x1_0',
        max_features_per_id=MAX_FEATURES_PER_ID,
        match_threshold=MATCH_THRESHOLD,
        spatial_distance_threshold=SPATIAL_DISTANCE_THRESHOLD
    )
    
    # Load YOLO model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n.pt')
    if os.path.exists(model_path):
        _yolo_model = YOLO(model_path)
    else:
        _yolo_model = YOLO("yolov8n.pt")
    
    # Unified loader for multi-camera
    _loader = get_loader('multi_camera', base_path=BASE_PATH, camera_folders=CAM_FOLDERS)
    _all_global_ids = set()
    
    _initialized = True
    print("[Multi-Cam-Multi] Initialized!")


def run_pipeline():
    """
    Process one frame from multi-camera dataset
    Returns: (grid_frame, stats) or raises StopIteration when dataset ends
    """
    _initialize()

    if getattr(_loader, "num_frames", 0) <= 0:
        raise RuntimeError(f"No frames found under {BASE_PATH}")
    
    try:
        frames, frame_idx = next(_loader)
    except StopIteration:
        # Reset loader and restart
        _loader.current_idx = 0
        _reid_manager.reset()
        _all_global_ids.clear()
        try:
            frames, frame_idx = next(_loader)
        except StopIteration as exc:
            raise RuntimeError("No frames available after reset in multi-camera loader") from exc
    
    props = _loader.get_properties()
    
    # Store all detections across cameras
    all_detections = []
    used_ids_per_camera = defaultdict(set)
    active_ids = set()
    
    # Process each camera
    for cam_id, frame in enumerate(frames):
        if frame is None:
            continue
        
        # Run YOLO detection with tracking
        results = _yolo_model.track(
            frame,
            persist=True,
            classes=[0],  # Person class
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # Process each detection
            for box in boxes:
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Use ReID manager to get global ID
                gid = _reid_manager.process(
                    frame=frame,
                    box=box,
                    cam_id=cam_id,
                    position=center,
                    frame_idx=frame_idx,
                    used_ids_in_frame=used_ids_per_camera
                )
                
                if gid is not None:
                    # Mark this ID as used in this camera
                    used_ids_per_camera[cam_id].add(gid)
                    active_ids.add(gid)
                    
                    all_detections.append({
                        'cam_id': cam_id,
                        'box': box,
                        'gid': gid
                    })
    
    # Visualize all cameras with detections
    processed = visualize_detections(frames, all_detections, frame_idx, props['num_frames'])
    
    # Create grid layout
    grid = create_grid_layout(processed)
    _all_global_ids.update(active_ids)

    stats_dict = {
        'active_ids': len(active_ids),
        'total_ids': len(_all_global_ids),
        'frame': frame_idx + 1,
        'heatmap': None,
        'map': None,
    }
    
    return grid, stats_dict


# Removed - now using unified loader


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
            color = get_color(gid)
            
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
    target_width = 640
    target_height = 360
    
    # Resize all frames to same size
    resized = []
    for f in frames:
        resized_frame = cv2.resize(f, (target_width, target_height))
        resized.append(resized_frame)
    
    if len(frames) == 2:
        # Side by side
        row = np.hstack(resized)
        return row
    
    elif len(frames) == 3:
        # 2 on top, 1 centered on bottom
        top_row = np.hstack([resized[0], resized[1]])
        
        # Center third camera
        padding_width = target_width // 2
        left_padding = np.zeros((target_height, padding_width, 3), dtype=np.uint8)
        right_padding = np.zeros((target_height, padding_width, 3), dtype=np.uint8)
        bottom_row = np.hstack([left_padding, resized[2], right_padding])
        
        grid = np.vstack([top_row, bottom_row])
        return grid
    
    elif len(frames) == 4:
        # 2x2 grid
        top_row = np.hstack(resized[:2])
        bottom_row = np.hstack(resized[2:4])
        grid = np.vstack([top_row, bottom_row])
        return grid
    
    else:
        # Rows of 3
        rows = []
        for i in range(0, len(frames), 3):
            row_frames = resized[i:i+3]
            while len(row_frames) < 3:
                row_frames.append(np.zeros((target_height, target_width, 3), dtype=np.uint8))
            row = np.hstack(row_frames)
            rows.append(row)
        return np.vstack(rows)


def main():
    """
    Standalone mode - display in OpenCV window
    """
    print("[Standalone Mode] Running multi_cam_multi_person.py")
    print("Press 'q' to quit")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    output_path = None
    
    while True:
        grid, stats = run_pipeline()
        
        # Initialize video writer
        if out is None:
            h, w = grid.shape[:2]
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'multi_cam_multi_person.mp4')
            out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))
            print(f"\nSaving output to: {output_path}")
        
        print(f"\rProcessing frame {stats['frame']} | Active IDs: {stats['active_ids']}", end="", flush=True)
        
        out.write(grid)
        cv2.imshow("Multi-Camera Person Tracking", grid)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n\nStopped by user")
            break
    
    print("\n\nProcessing complete!")
    print(f"Total unique persons tracked: {stats['total_ids']}")
    print(f"Output saved to: {output_path}")
    
    if out:
        out.release()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
