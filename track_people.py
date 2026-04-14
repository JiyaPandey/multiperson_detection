"""
Minimal Person Tracking Script (Fixed & Stable)
YOLOv8 + ByteTrack
"""

import cv2
from ultralytics import YOLO


def track_people(video_path, output_path=None, confidence=0.5):
    # Load model
    model = YOLO('yolov8n.pt')

    # Open video (FIXED)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps}fps")
    print("Press 'q' to quit\n")

    # Video writer (FIXED extension)
    writer = None
    if output_path:
        if not output_path.endswith(".mp4"):
            output_path += ".mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    track_ids = []  # prevent crash

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # YOLO + ByteTrack
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=confidence,
                verbose=False
            )

            track_ids = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)

                    # Bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label (IMPROVED)
                    label = f"ID:{track_id} {conf:.2f}"

                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_h - 10),
                        (x1 + label_w + 10, y1),
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )

            # Show frame
            cv2.imshow('Person Tracking', frame)

            # Save output
            if writer:
                writer.write(frame)

            # Progress print
            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | Tracked: {len(track_ids)}")

            # KEY FIX (window stability)
            key = cv2.waitKey(25)
            if key == ord('q'):
                print("\nStopped by user")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()

        if writer:
            writer.release()
            print(f"\nOutput saved to: {output_path}")

        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "C:\\Users\\HP\\Downloads\\TUD-Stadtmitte-raw.webm"

    output_path = (
        "C:\\Users\\HP\\Desktop\\multiperson_detection\\output.mp4"
        if len(sys.argv) > 2 else None
    )

    print("=" * 60)
    print("Person Tracking - YOLOv8 + ByteTrack")
    print("=" * 60)

    track_people(video_path, output_path)