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


def _initialize():
    """Initialize global state (called once)"""
    global _model, _loader, _tracking_data, _heatmap, _id_last_seen, _event_log, _all_ids_seen, _initialized
    
    if _initialized:
        return
    
    video_path = r"C:\Users\HP\Downloads\TUD-Stadtmitte-raw.webm"
    data_video = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.mp4')
    if os.path.exists(data_video):
        video_path = data_video
    
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
    
    # Labels
    cv2.putText(map_img, f"2D MAP | People: {len(current_ids)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    cv2.putText(heatmap_img, "HEATMAP", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Stack panels
    right_panel = np.vstack((map_img, heatmap_img))
    
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
    
    # Stats panel
    stats = np.ones((stats_height, width + map_width, 3), dtype=np.uint8) * 20
    
    cv2.putText(stats, "REAL-TIME ANALYTICS", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
    cv2.putText(stats, f"Active: {len(current_ids)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)
    cv2.putText(stats, f"IDs: {[int(i) for i in current_ids]}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    
    # Traffic warning
    if len(current_ids) > 8:
        cv2.putText(frame, "HIGH TRAFFIC", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Event log
    log_x = width + 10
    cv2.putText(stats, "Events:", (log_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    
    for i, e in enumerate(_event_log):
        color = (0, 255, 0) if "ENTERED" in e else (0, 0, 255)
        cv2.putText(stats, e, (log_x, 70 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # Combine dashboard
    top = np.hstack((frame, right_panel))
    dashboard = np.vstack((top, stats))
    
    # Scale if needed
    if dashboard.shape[1] > 1920:
        scale = 1920 / dashboard.shape[1]
        dashboard = cv2.resize(dashboard, (0, 0), fx=scale, fy=scale)
    
    stats_dict = {
        'active_ids': len(current_ids),
        'total_ids': len(_all_ids_seen),
        'heatmap': heatmap_img,
        'map': map_img,
        'frame': _frame_count,
    }

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
    # Default video path
    video_path = r"C:\Users\HP\Downloads\TUD-Stadtmitte-raw.webm"
    
    # Check if video exists in data folder
    data_video = os.path.join('..', 'data', 'input.mp4')
    if os.path.exists(data_video):
        video_path = data_video
    
    track_people(video_path, confidence=0.5, max_history=12)
