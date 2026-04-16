import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine
from collections import deque

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8m.pt"
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

# ---------------- INIT ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO(MODEL_PATH)
reid_model = FeatureExtractor(model_name='osnet_x1_0', device=device)

global_db = {}          # gid -> deque([feature, ...])
local_to_global = {}    # (cam_id, local_track_id) -> gid
id_confidence = {}      # gid -> confidence score
last_seen = {}          # gid -> (cam_id, frame_idx)
last_reid_frame = {}    # (cam_id, local_track_id) -> last frame idx
next_global_id = 1

# ---------------- PREPROCESS ----------------
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced

# ---------------- FEATURE ----------------
def normalize_feature(feat):
    norm = np.linalg.norm(feat)
    if norm == 0:
        return None
    return feat / norm

def extract_feature(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (128, 256))
    feat = reid_model([crop])[0].cpu().numpy()
    return normalize_feature(feat)

def average_cosine_distance(feature, feature_list):
    if not feature_list:
        return float('inf')
    return float(np.mean([cosine(feature, f) for f in feature_list]))

def update_global_db(gid, feature):
    if gid not in global_db:
        global_db[gid] = deque(maxlen=MAX_FEATURES)
    global_db[gid].append(feature)

def is_good_crop(box, conf, frame_shape):
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
    # if (bh / max(bw, 1)) < 1.2:
    #     return False
    return True

# ---------------- GLOBAL ID ----------------
def find_best_gid(feature, gids):
    best_gid = None
    best_dist = float('inf')
    for gid in gids:
        dist = average_cosine_distance(feature, global_db.get(gid, []))
        if dist < best_dist:
            best_dist = dist
            best_gid = gid
    return best_gid, best_dist

def create_new_gid(feature, cam_id, frame_idx):
    global next_global_id
    gid = next_global_id
    next_global_id += 1
    update_global_db(gid, feature)
    id_confidence[gid] = 0.3
    last_seen[gid] = (cam_id, frame_idx)
    return gid

def update_confidence(gid, matched):
    current = id_confidence.get(gid, 0.0)
    if matched:
        id_confidence[gid] = min(1.0, current + CONFIDENCE_INC)
    else:
        id_confidence[gid] = max(0.0, current - CONFIDENCE_DECAY)

def select_gid(feature, cam_id, frame_idx, current_gid=None):
    same_cam_gids = [gid for gid, (cam, _) in last_seen.items() if cam == cam_id]
    best_gid, best_dist = find_best_gid(feature, same_cam_gids)

    if best_gid is None or best_dist >= MATCH_THRESHOLD:
        best_gid, best_dist = find_best_gid(feature, global_db.keys())

    if current_gid is not None:
        current_dist = average_cosine_distance(feature, global_db.get(current_gid, []))
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

# ---------------- MAIN ----------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video")
        return

    frame_count = 0
    last_results = [None, None, None, None]

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        h, w, _ = frame.shape

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
            proc_cam = preprocess(cam)

            if frame_count % FRAME_SKIP == 0:
                results = yolo_model.track(
                    proc_cam,
                    persist=True,
                    classes=[0],
                    conf=0.15,
                    iou=0.5,
                    imgsz=960,
                    verbose=False
                )

                if results is None or results[0].boxes is None:
                    results = yolo_model(
                        proc_cam,
                        conf=0.15,
                        imgsz=960,
                        verbose=False
                    )

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
                    if tid is None:
                        key = (i, int(x1 / 10), int(y1 / 10))
                    else:
                        key = (i, tid)

                    current_gid = local_to_global.get(key)

                    # Always assign an ID if it does not exist yet.
                    if current_gid is None:

                        feat = extract_feature(cam, box)

                        # Case 1: Feature works -> normal ReID.
                        if feat is not None:
                            selected_gid = select_gid(feat, i, frame_count, current_gid=None)

                            if selected_gid is None:
                                selected_gid = create_new_gid(feat, i, frame_count)

                            update_global_db(selected_gid, feat)

                        # Case 2: Feature fails -> fallback ID.
                        else:
                            global next_global_id
                            selected_gid = next_global_id
                            next_global_id += 1

                            global_db[selected_gid] = deque(maxlen=MAX_FEATURES)
                            id_confidence[selected_gid] = 0.3
                            last_seen[selected_gid] = (i, frame_count)

                        # Always assign.
                        local_to_global[key] = selected_gid
                        last_reid_frame[key] = frame_count
                        update_confidence(selected_gid, matched=True)
                        matched_gids.add(selected_gid)

                    # Update existing ID (only if feature works).
                    else:
                        reid_ready = (frame_count - last_reid_frame.get(key, -REID_INTERVAL)) >= REID_INTERVAL

                        if reid_ready:
                            feat = extract_feature(cam, box)

                            if feat is not None:
                                selected_gid = select_gid(feat, i, frame_count, current_gid=current_gid)

                                local_to_global[key] = selected_gid
                                update_global_db(selected_gid, feat)
                                last_seen[selected_gid] = (i, frame_count)
                                last_reid_frame[key] = frame_count
                                update_confidence(selected_gid, matched=True)
                                matched_gids.add(selected_gid)

                    if key in local_to_global:
                        gid = local_to_global[key]
                        label = f"GID {gid}"
                        color = (0, 255, 0)
                    else:
                        label = f"TID {tid}" if tid is not None else "DET"
                        color = (0, 0, 255)

                    cv2.rectangle(cam_display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(cam_display, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(cam_display, f"CAM {i+1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            processed.append(cam_display)

        for gid in list(id_confidence.keys()):
            if gid not in matched_gids:
                update_confidence(gid, matched=False)

        grid = np.vstack((
            np.hstack((processed[0], processed[1])),
            np.hstack((processed[2], processed[3]))
        ))

        cv2.imshow("Multi-Camera Tracking", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()