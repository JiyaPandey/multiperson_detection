import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque


def track_people(video_path, confidence=0.5, max_history=12):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracking_data = defaultdict(lambda: deque(maxlen=max_history))

    map_width = 320
    map_height = height // 2
    stats_height = 200

    heatmap = np.zeros((map_height, map_width), dtype=np.float32)

    id_last_seen = {}
    event_log = deque(maxlen=6)

    id_colors = {}
    def get_color(track_id):
        if track_id not in id_colors:
            np.random.seed(track_id)
            id_colors[track_id] = tuple(map(int, np.random.randint(80, 255, 3)))
        return id_colors[track_id]

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        current_ids = set()

        map_img = np.ones((map_height, map_width, 3), dtype=np.uint8) * 30

        results = model.track(frame, persist=True, classes=[0], conf=confidence, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1

                if h < 0.15 * height or w < 0.05 * width:
                    continue

                cx = x1 + w // 2
                foot_y = y2

                current_ids.add(track_id)
                color = get_color(track_id)

                map_x = int((cx / width) * map_width)
                map_y = int((foot_y / height) * map_height)

                tracking_data[track_id].append((map_x, map_y))

                # Heatmap accumulation (balanced)
                if 0 <= map_x < map_width and 0 <= map_y < map_height:
                    heatmap[map_y, map_x] += 8.0

                # Bounding box
                label = f"ID {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                pts = list(tracking_data[track_id])

                # Flow arrows
                if len(pts) >= 2:
                    cv2.arrowedLine(map_img, pts[-2], pts[-1],
                                    color, 2, tipLength=0.4)

                if len(pts) > 1:
                    cv2.polylines(map_img, [np.array(pts)], False, color, 1)

                cv2.circle(map_img, (map_x, map_y), 4, color, -1)

        # Smooth decay (slow)
        heatmap *= 0.995

        # Heatmap rendering
        heatmap_img = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

        if np.max(heatmap) > 0:
            blur = cv2.GaussianBlur(heatmap, (31, 31), 0)
            norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

            # Remove weak noise
            color[norm < 30] = [255, 255, 255]

            heatmap_img = color

        # Labels
        cv2.putText(map_img, f"2D MAP | People: {len(current_ids)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        cv2.putText(heatmap_img, "HEATMAP", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        heatmap_img = cv2.resize(heatmap_img, (map_width, height - map_height))

        right_panel = np.vstack((map_img, heatmap_img))

        # Events
        for i in current_ids:
            if i not in id_last_seen:
                event_log.appendleft(f"ID {i} ENTERED")
            id_last_seen[i] = frame_count

        expired = []
        for i, last in id_last_seen.items():
            if i not in current_ids and (frame_count - last) > fps:
                event_log.appendleft(f"ID {i} LEFT")
                expired.append(i)

        for i in expired:
            del id_last_seen[i]

        # Stats panel
        stats = np.ones((stats_height, width + map_width, 3), dtype=np.uint8) * 20

        cv2.putText(stats, "REAL-TIME ANALYTICS",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

        cv2.putText(stats, f"Active: {len(current_ids)}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)

        cv2.putText(stats, f"IDs: {[int(i) for i in current_ids]}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        # Smarter congestion threshold
        if len(current_ids) > 8:
            cv2.putText(frame, "HIGH TRAFFIC", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Event log
        log_x = width + 10
        cv2.putText(stats, "Events:", (log_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        for i, e in enumerate(event_log):
            color = (0, 255, 0) if "ENTERED" in e else (0, 0, 255)
            cv2.putText(stats, e, (log_x, 70 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Combine UI
        top = np.hstack((frame, right_panel))
        dashboard = np.vstack((top, stats))

        if dashboard.shape[1] > 1920:
            scale = 1920 / dashboard.shape[1]
            dashboard = cv2.resize(dashboard, (0, 0), fx=scale, fy=scale)

        cv2.imshow("Analytics Dashboard", dashboard)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_people("C:\\Users\\HP\\Downloads\\TUD-Stadtmitte-raw.webm")