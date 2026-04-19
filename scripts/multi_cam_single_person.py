"""
MULTI CAMERA (GRID) - SINGLE PERSON TRACKING
Splits video into grid, tracks same person across multiple cameras using ReID
"""

print("Running: MULTI CAM SAME PERSON")

import cv2
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine
from collections import deque
from src.reid.reid_manager import ReIDManager
from src.utils.visualization import get_color, draw_bbox, draw_label
from src.input.video_input import get_loader


# Config
MODEL_PATH = os.path.join('..', 'models', 'yolov8n.pt')
VIDEO_PATH = r"C:\Users\HP\Downloads\CCTV_Camera_Effect_-_Adobe_After_Effects_720p.mp4"

FRAME_SKIP = 2
REID_INTERVAL = 5
MATCH_THRESHOLD = 0.65
MAX_FEATURES = 10
REID_CONF_MIN = 0.25
MIN_BOX_AREA_RATIO = 0.003
MIN_BOX_SIZE = 40
EDGE_MARGIN = 4

CONFIDENCE_INC = 0.08
CONFIDENCE_DECAY = 0.01
CONFIDENCE_ACCEPT = 0.4
SWITCH_MARGIN = 0.08


# Global tracking data (for local track to global ID mapping)
local_to_global = {}    # (cam_id, local_track_id) -> gid
id_confidence = {}      # gid -> confidence score
last_seen = {}          # gid -> (cam_id, frame_idx)
last_reid_frame = {}    # (cam_id, local_track_id) -> last frame idx
next_global_id = 1


def preprocess_gray(frame):
    """Enhance grayscale frames"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


def preprocess_color(frame):
    """Enhance color frames"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def is_grayscale(frame):
    """Check if frame is grayscale"""
    return np.mean(np.abs(frame[:, :, 0] - frame[:, :, 1])) < 2


def is_good_crop(box, conf, frame_shape):
    """Check if bounding box is good for ReID feature extraction"""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    
    if conf < REID_CONF_MIN:
        return False
    if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
        return False
    if (bw * bh) < (MIN_BOX_AREA_RATIO * w * h):
        return False
    if x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= (w - EDGE_MARGIN) or y2 >= (h - EDGE_MARGIN):
        return False
    
    return True


def average_cosine_distance(feature, feature_list):
    """Compute average cosine distance to feature list"""
    if not feature_list:
        return float('inf')
    return float(np.mean([cosine(feature, f) for f in feature_list]))


def update_global_db(reid_manager, gid, feature):
    """Update global ReID database"""
    if feature is not None:
        reid_manager.global_id_features[gid].append(feature)


def find_best_gid(reid_manager, feature, gids):
    """Find best matching global ID"""
    best_gid = None
    best_dist = float('inf')
    for gid in gids:
        features = reid_manager.global_id_features.get(gid, [])
        dist = average_cosine_distance(feature, features)
        if dist < best_dist:
            best_dist = dist
            best_gid = gid
    return best_gid, best_dist


def create_new_gid(reid_manager, feature, cam_id, frame_idx):
    """Create new global ID"""
    global next_global_id
    gid = next_global_id
    next_global_id += 1
    update_global_db(reid_manager, gid, feature)
    id_confidence[gid] = 0.3
    last_seen[gid] = (cam_id, frame_idx)
    return gid


def update_confidence(gid, matched):
    """Update ID confidence score"""
    current = id_confidence.get(gid, 0.0)
    if matched:
        id_confidence[gid] = min(1.0, current + CONFIDENCE_INC)
    else:
        id_confidence[gid] = max(0.0, current - CONFIDENCE_DECAY)


def select_gid(reid_manager, feature, cam_id, frame_idx, current_gid=None):
    """Select best global ID for a feature"""
    same_cam_gids = [gid for gid, (cam, _) in last_seen.items() if cam == cam_id]
    best_gid, best_dist = find_best_gid(reid_manager, feature, same_cam_gids)

    if best_gid is None or best_dist >= MATCH_THRESHOLD:
        best_gid, best_dist = find_best_gid(reid_manager, feature, reid_manager.global_id_features.keys())

    if current_gid is not None:
        current_features = reid_manager.global_id_features.get(current_gid, [])
        current_dist = average_cosine_distance(feature, current_features)
        if best_gid == current_gid:
            return current_gid
        if best_gid is not None and best_dist < MATCH_THRESHOLD:
            if id_confidence.get(best_gid, 0.0) >= 0.2:
                if current_dist == float('inf') or (best_dist + SWITCH_MARGIN) < current_dist:
                    return best_gid
        return current_gid

    if best_gid is not None and best_dist < MATCH_THRESHOLD:
        return best_gid
    return None


def main():
    global next_global_id
    
    print(f"Loading video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    
    # Initialize ReID manager
    reid_manager = ReIDManager(
        model_name='osnet_x1_0',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_features_per_id=MAX_FEATURES,
        match_threshold=MATCH_THRESHOLD
    )
    
    # Load YOLO model
    yolo_model = YOLO(MODEL_PATH)
    
    # Unified loader
    loader = get_loader('video', video_path=VIDEO_PATH)
    props = loader.get_properties()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_count = 0
    last_results = [None, None, None, None]

    print("Processing... Press 'q' to quit")

    for frame in loader:
        frame_count += 1
        h, w, _ = frame.shape

        # Split frame into 4 cameras (2x2 grid)
        cams = [
            frame[0:h//2, 0:w//2],
            frame[0:h//2, w//2:w],
            frame[h//2:h, 0:w//2],
            frame[h//2:h, w//2:w]
        ]

        processed = []
        matched_gids = set()

        for i, cam in enumerate(cams):
            cam_display = cam.copy()
            
            # Preprocess based on frame type
            if is_grayscale(cam):
                proc_cam = preprocess_gray(cam)
            else:
                proc_cam = preprocess_color(cam)

            # Run detection/tracking
            if frame_count % FRAME_SKIP == 0:
                results = yolo_model.track(
                    proc_cam,
                    persist=True,
                    classes=[0],
                    conf=0.15,
                    iou=0.5,
                    imgsz=640,
                    verbose=False
                )

                if results is None or results[0].boxes is None:
                    results = yolo_model(proc_cam, conf=0.15, imgsz=640, verbose=False)

                last_results[i] = results
            else:
                results = last_results[i]

            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(boxes)

                for box, tid, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Create key for local-to-global mapping
                    if tid is None:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        key = (i, cx // 20, cy // 20)
                    else:
                        key = (i, tid)

                    current_gid = local_to_global.get(key)

                    # Assign new ID
                    if current_gid is None:
                        if not is_good_crop(box, conf, cam.shape):
                            continue

                        feat = reid_manager.extract_feature(cam, box)

                        if feat is not None:
                            selected_gid = select_gid(reid_manager, feat, i, frame_count, current_gid=None)

                            if selected_gid is None:
                                selected_gid = create_new_gid(reid_manager, feat, i, frame_count)

                            update_global_db(reid_manager, selected_gid, feat)
                        else:
                            # Fallback ID
                            selected_gid = next_global_id
                            next_global_id += 1
                            reid_manager.global_id_features[selected_gid] = deque(maxlen=MAX_FEATURES)
                            id_confidence[selected_gid] = 0.3
                            last_seen[selected_gid] = (i, frame_count)

                        local_to_global[key] = selected_gid
                        last_reid_frame[key] = frame_count
                        update_confidence(selected_gid, matched=True)
                        matched_gids.add(selected_gid)

                    # Update existing ID
                    else:
                        reid_ready = (frame_count - last_reid_frame.get(key, -REID_INTERVAL)) >= REID_INTERVAL

                        if reid_ready:
                            if not is_good_crop(box, conf, cam.shape):
                                continue

                            feat = reid_manager.extract_feature(cam, box)

                            if feat is not None:
                                selected_gid = select_gid(reid_manager, feat, i, frame_count, current_gid=current_gid)

                                local_to_global[key] = selected_gid
                                update_global_db(reid_manager, selected_gid, feat)
                                last_seen[selected_gid] = (i, frame_count)
                                last_reid_frame[key] = frame_count
                                update_confidence(selected_gid, matched=True)
                                matched_gids.add(selected_gid)

                    # Draw visualization
                    if key in local_to_global:
                        gid = local_to_global[key]
                        label = f"GID {gid}"
                        color = get_color(gid)
                    else:
                        label = f"TID {tid}" if tid is not None else "DET"
                        color = (0, 0, 255)

                    cv2.rectangle(cam_display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(cam_display, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Camera label
            cv2.putText(cam_display, f"CAM {i+1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            processed.append(cam_display)

        # Update confidence for non-matched IDs
        for gid in list(id_confidence.keys()):
            if gid not in matched_gids:
                update_confidence(gid, matched=False)

        # Create 2x2 grid
        grid = np.vstack((
            np.hstack((processed[0], processed[1])),
            np.hstack((processed[2], processed[3]))
        ))

        # Initialize video writer
        if out is None:
            output_dir = os.path.join('..', 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'multi_cam_same_person.mp4')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (grid.shape[1], grid.shape[0]))

        out.write(grid)
        cv2.imshow("Multi-Camera Tracking", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    loader.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print("\nTracking complete!")
    print(f"Total unique persons: {next_global_id - 1}")


if __name__ == "__main__":
    main()
