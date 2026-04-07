# detector.py
# Vehicle detection using YOLOv8.
# Supports two modes:
#   1. COCO pre-trained model (default) — filters for vehicle classes only
#   2. Custom-trained model (best_parking.pt) — trained on Parking-AMU50 dataset
# Automatically uses the custom model if best_parking.pt exists in the project folder.
# Supports runtime confidence threshold adjustment from the dashboard.

import os
from ultralytics import YOLO


# Path to the custom-trained model (produced by train.py)
CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_parking.pt")


class VehicleDetector:
    """
    Wraps a YOLOv8 model to detect vehicles in individual video frames.

    If a custom-trained model (best_parking.pt) exists, it is used automatically.
    Otherwise, falls back to the COCO pre-trained YOLOv8n model with vehicle
    class filtering.
    """

    # COCO class IDs that correspond to vehicles (used only with COCO model)
    COCO_VEHICLE_CLASS_IDS = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.40):
        """
        Load the YOLOv8 model.

        If best_parking.pt exists in the project folder, it is used instead
        of the provided model_path. The custom model was trained specifically
        on parking violation data and detects all relevant classes directly.

        Args:
            model_path (str): Path to the fallback YOLOv8 weights file.
                              'yolov8n.pt' (nano) will be auto-downloaded on first run.
            confidence_threshold (float): Minimum confidence score to keep a detection.
        """
        # Auto-detect custom trained model
        self.using_custom_model = False
        if os.path.exists(CUSTOM_MODEL_PATH):
            model_path = CUSTOM_MODEL_PATH
            self.using_custom_model = True
            print(f"[Detector] ✅ Custom model found: {CUSTOM_MODEL_PATH}")
        else:
            print(f"[Detector] Using default COCO model: {model_path}")
            print(f"[Detector] (Run 'python train.py' to train a custom model)")

        print(f"[Detector] Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Read class names from the custom model
        if self.using_custom_model:
            self.custom_class_names = self.model.names  # dict {id: name}
            print(f"[Detector] Custom model classes: {self.custom_class_names}")
        else:
            self.custom_class_names = None

        print("[Detector] Model loaded successfully.")

    def set_confidence(self, value):
        """
        Update the confidence threshold at runtime.

        Args:
            value (float): New threshold between 0.0 and 1.0.
        """
        self.confidence_threshold = max(0.05, min(0.95, value))

    def detect(self, frame):
        """
        Run vehicle detection on a single frame.

        If using the custom model, ALL detected classes are returned
        (since the model was trained specifically on parking data).
        If using the COCO model, only vehicle classes are returned.

        Args:
            frame (np.ndarray): BGR image frame from OpenCV.

        Returns:
            list of tuples: Each tuple is (x1, y1, x2, y2, class_name, confidence).
                            Coordinates are integers in pixel space.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < self.confidence_threshold:
                continue

            if self.using_custom_model:
                # Custom model: accept all classes (trained specifically for parking)
                class_name = self.custom_class_names.get(cls_id, "vehicle")
            else:
                # COCO model: filter for vehicle classes only
                if cls_id not in self.COCO_VEHICLE_CLASS_IDS:
                    continue
                class_name = self.COCO_VEHICLE_CLASS_IDS[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, class_name, conf))

        return detections
