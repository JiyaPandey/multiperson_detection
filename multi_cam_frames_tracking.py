import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine
from collections import deque

# ---------------- CONFIG ----------------
BASE_PATH = "EPFL_RLC_dataset/frames"
CAM_FOLDERS = ["cam0", "cam1", "cam2"]

FRAME_SKIP = 2
MATCH_THRESHOLD = 0.65
MAX_FEATURES = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

yolo_model = YOLO("yolov8n.pt")
reid_model = FeatureExtractor(model_name='osnet_x1_0', device=device)

# ---------------- GLOBAL STORAGE ----------------
global_db = {}
local_to_global = {}
next_gid = 1

# ---------------- UTILS ----------------
def normalize(feat):
    norm = np.linalg.norm(feat)
    return feat / norm if norm != 0 else None

def extract_feature(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (128, 256))
    feat = reid_model([crop])[0].cpu().numpy()
    return normalize(feat)

def get_gid(feature):
    global next_gid

    best_gid = None
    best_dist = 999

    for gid, feats in global_db.items():
        dists = [cosine(feature, f) for f in feats]
        dist = np.mean(dists)
        if dist < best_dist:
            best_dist = dist
            best_gid = gid

    if best_dist < MATCH_THRESHOLD:
        global_db[best_gid].append(feature)
        return best_gid
    else:
        global_db[next_gid] = deque([feature], maxlen=MAX_FEATURES)
        next_gid += 1
        return next_gid - 1

# ---------------- LOAD FRAMES ----------------
def load_camera_frames():
    cams = []
    for cam in CAM_FOLDERS:
        path = os.path.join(BASE_PATH, cam)
        frames = sorted(os.listdir(path))
        frames = [os.path.join(path, f) for f in frames if f.endswith((".jpg", ".png"))]
        cams.append(frames)
    return cams

# ---------------- MAIN ----------------
def main():
    cams = load_camera_frames()
    min_len = min(len(cam) for cam in cams)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    for i in range(min_len):
        frames = [cv2.imread(cam[i]) for cam in cams]

        processed = []

        for cam_id, frame in enumerate(frames):
            display = frame.copy()

            results = yolo_model.track(
                frame,
                persist=True,
                classes=[0],
                conf=0.3,
                imgsz=640,
                verbose=False
            )

            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)

                    # center-based key
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    key = (cam_id, cx // 20, cy // 20)

                    if key not in local_to_global:
                        feat = extract_feature(frame, box)
                        if feat is not None:
                            gid = get_gid(feat)
                            local_to_global[key] = gid

                    if key in local_to_global:
                        gid = local_to_global[key]
                        color = (0, 255, 0)
                        label = f"GID {gid}"
                    else:
                        color = (0, 0, 255)
                        label = "DET"

                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(display, f"CAM {cam_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            processed.append(display)

        # create grid
        grid = np.hstack(processed)

        if out is None:
            out = cv2.VideoWriter("output_frames.mp4", fourcc, 20, (grid.shape[1], grid.shape[0]))

        out.write(grid)

        cv2.imshow("Multi-Cam Frames", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()