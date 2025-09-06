import cv2
import datetime
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEO_PATH = "../data/demo_video.mp4"  # your video file
DETECTOR_WEIGHTS = "../models/yolov8n.pt"  # YOLOv8 tiny pretrained
LOG_FILE = "../data/log.csv"
LINE_POSITION = 300  # Y-coordinate for counting line
CONFIDENCE_THRESHOLD = 0.3

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def log_row(file_path, row):
    """Append a row to CSV."""
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# -----------------------------
# DETECTOR
# -----------------------------
class Detector:
    def __init__(self, weights):
        self.model = YOLO(weights)

    def detect(self, frame):
        results = self.model(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD)[0]
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if cls.item() == 0:  # class 0 = person in COCO
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2, y2], conf.item(), "person"))
        return detections

# -----------------------------
# TRACKER
# -----------------------------
class Tracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame=None):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()
            track_id = t.track_id
            results.append((track_id, ltrb))
        return results

# -----------------------------
# COUNTER
# -----------------------------
class LineCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.last_centroid = {}  # track_id -> last y position
        self.count_in = 0
        self.count_out = 0

    def update(self, tracks):
        for tid, (x1, y1, x2, y2) in tracks:
            cy = int((y1 + y2) / 2)
            if tid in self.last_centroid:
                prev_y = self.last_centroid[tid]
                if prev_y < self.line_y <= cy:  # entering
                    self.count_in += 1
                elif prev_y > self.line_y >= cy:  # exiting
                    self.count_out += 1
            self.last_centroid[tid] = cy
        return self.count_in, self.count_out

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    # Initialize
    detector = Detector(DETECTOR_WEIGHTS)
    tracker = Tracker()
    counter = LineCounter(LINE_POSITION)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        return

    # Create CSV log if not exists
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "people_in", "people_out", "occupancy"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # DETECTION
        detections = detector.detect(frame)

        # TRACKING
        tracks = tracker.update(detections, frame=frame)
        track_boxes = [ltrb for tid, ltrb in tracks]

        # COUNTING
        in_count, out_count = counter.update(tracks)
        occupancy = max(0, in_count - out_count)

        # LOGGING
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_row(LOG_FILE, [timestamp, in_count, out_count, occupancy])

        # -----------------------------
        # VISUALIZATION
        # -----------------------------
        # Draw counting line
        cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 0, 255), 2)

        # Draw tracked boxes
        for tid, (x1, y1, x2, y2) in tracks:
            # Convert coordinates to int
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Display counts
        cv2.putText(frame, f"In: {in_count} Out: {out_count} Occupancy: {occupancy}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Retail Analytics", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished. Logs saved to", LOG_FILE)

if __name__ == "__main__":
    main()
