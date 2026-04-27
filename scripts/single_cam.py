"""
SINGLE CAMERA - MULTI PERSON TRACKING
Uses YOLO tracking with analytics dashboard
"""

print("Running: SINGLE CAM MULTI PERSON")

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ultralytics import YOLO
from collections import defaultdict, deque
from src.utils.visualization import get_color, draw_bbox, draw_label
from src.analytics.heatmap import Heatmap
from src.input.video_input import get_loader


# =========================
# GLOBAL STATE FOR DASHBOARD
# =========================
_model = None
_loader = None
_tracking_data = None
_heatmap = None
_id_last_seen = None
_event_log = None
_all_ids_seen = None
_frame_count = 0
_initialized = False


def _resolve_video_path():
    """Resolve a valid input video path for local and container runs."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    env_video = os.getenv('SINGLE_CAM_VIDEO_PATH')
    candidates = [
        env_video,
        os.path.join(data_dir, 'TUD-Stadtmitte-raw.webm'),
        os.path.join(data_dir, 'input.mp4'),
        os.path.join(data_dir, 'input.webm'),
        os.path.join(data_dir, 'CCTV_Camera_Effect_-_Adobe_After_Effects_720p.mp4'),
        os.path.join(data_dir, 'ADL-Rundle-6-raw.webm'),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv')):
                candidate = os.path.join(data_dir, name)
                if os.path.exists(candidate):
                    return candidate

    raise FileNotFoundError(
        "No readable video found for single-cam mode. "
        "Set SINGLE_CAM_VIDEO_PATH, or place a video file under data/"
    )


def _initialize():
    """Initialize global state (called once)"""
    global _model, _loader, _tracking_data, _heatmap, _id_last_seen, _event_log, _all_ids_seen, _initialized
    
    if _initialized:
        return
    
    video_path = _resolve_video_path()
    
    print(f"[Single-Cam] Initializing with video: {video_path}")
    
    _model = YOLO(os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n.pt'))
    _loader = get_loader('video', video_path=video_path)
    
    props = _loader.get_properties()
    map_width = 320
    map_height = props['height'] // 2
    
    _tracking_data = defaultdict(lambda: deque(maxlen=12))
    _heatmap = Heatmap(map_width, map_height, decay_factor=0.995, blur_kernel=31, weight=8.0)
    _id_last_seen = {}
    _event_log = deque(maxlen=6)
    _all_ids_seen = set()
    
    _initialized = True
    print("[Single-Cam] Initialized!")


def run_pipeline():
    """
    Process one frame and return visualization
    Returns: (frame, stats) or None if video ended
    """
    global _model, _loader, _tracking_data, _heatmap, _id_last_seen, _event_log, _all_ids_seen, _frame_count
    
    _initialize()
    
    try:
        frame = next(_loader)
    except StopIteration:
        # Reset and restart
        _loader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = next(_loader)
        _all_ids_seen.clear()
        _frame_count = 0
    
    _frame_count += 1
    height, width = frame.shape[:2]
    map_width = 320
    map_height = height // 2
    stats_height = 200
    
    current_ids = set()
    
    # Create 2D map
    map_img = np.ones((map_height, map_width, 3), dtype=np.uint8) * 30
    
    # Run YOLO tracking
    results = _model.track(frame, persist=True, classes=[0], conf=0.5, verbose=False)
    
    detections_for_heatmap = []
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            # Filter small detections
            if h < 0.15 * height or w < 0.05 * width:
                continue
            
            cx = x1 + w // 2
            foot_y = y2
            
            current_ids.add(track_id)
            color = get_color(track_id)
            
            # Map coordinates
            map_x = int((cx / width) * map_width)
            map_y = int((foot_y / height) * map_height)
            
            # Store trajectory
            _tracking_data[track_id].append((map_x, map_y))
            detections_for_heatmap.append((map_x, map_y))
            
            # Draw bounding box on frame
            draw_bbox(frame, box, track_id, color)
            
            # Draw on 2D map
            pts = list(_tracking_data[track_id])
            
            if len(pts) >= 2:
                cv2.arrowedLine(map_img, pts[-2], pts[-1], color, 2, tipLength=0.4)
            
            if len(pts) > 1:
                cv2.polylines(map_img, [np.array(pts)], False, color, 1)
            
            cv2.circle(map_img, (map_x, map_y), 4, color, -1)
    
    # Update heatmap
    _heatmap.update(detections_for_heatmap)
    heatmap_img = _heatmap.render()
    heatmap_img = cv2.resize(heatmap_img, (map_width, height - map_height))

    heatmap_img = cv2.normalize(heatmap_img, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_img = heatmap_img.astype(np.uint8)

    if len(heatmap_img.shape) == 2:
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

    map_img = map_img.astype(np.uint8)

    if len(map_img.shape) == 2:
        map_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    
    # Labels
    cv2.putText(map_img, f"2D MAP | People: {len(current_ids)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    cv2.putText(heatmap_img, "HEATMAP", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Event detection
    fps = _loader.get_properties()['fps']
    for i in current_ids:
        if i not in _id_last_seen:
            _event_log.appendleft(f"ID {i} ENTERED")
        _id_last_seen[i] = _frame_count

    _all_ids_seen.update(current_ids)
    
    expired = []
    for i, last in _id_last_seen.items():
        if i not in current_ids and (_frame_count - last) > fps:
            _event_log.appendleft(f"ID {i} LEFT")
            expired.append(i)
    
    for i in expired:
        del _id_last_seen[i]
    
    # Traffic warning
    if len(current_ids) > 8:
        cv2.putText(frame, "HIGH TRAFFIC", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    active_ids = current_ids
    total_ids = len(_all_ids_seen)
    frame_count = _frame_count
    stats_dict = {
        "heatmap": heatmap_img,
        "map": map_img,
        "active_ids": len(active_ids),
        "total_ids": total_ids,
        "frame": frame_count
    }

    # Return only the camera frame to keep Streamlit layout clean.
    return frame, stats_dict


def track_people(video_path, confidence=0.5, max_history=12):
    """
    Single camera person tracking with real-time analytics (standalone mode)
    """
    print(f"[Standalone Mode] Running single_cam.py with OpenCV window")
    print("Press 'q' to quit")
    
    while True:
        frame, stats = run_pipeline()
        map_img = stats.get('map')
        heatmap_img = stats.get('heatmap')

        if map_img is None:
            map_img = np.zeros((frame.shape[0] // 2, 320, 3), dtype=np.uint8)
        if heatmap_img is None:
            heatmap_img = np.zeros((frame.shape[0] - (frame.shape[0] // 2), 320, 3), dtype=np.uint8)

        right_panel = np.vstack((map_img, heatmap_img))
        stats_panel = np.ones((200, frame.shape[1] + right_panel.shape[1], 3), dtype=np.uint8) * 20
        cv2.putText(stats_panel, f"Active: {stats.get('active_ids', 0)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 150), 2)
        cv2.putText(stats_panel, f"Total IDs: {stats.get('total_ids', 0)}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

        top = np.hstack((frame, right_panel))
        dashboard = np.vstack((top, stats_panel))

        cv2.imshow("Analytics Dashboard", dashboard)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("\nTracking complete!")


if __name__ == "__main__":
    video_path = _resolve_video_path()
    track_people(video_path, confidence=0.5, max_history=12)
