# evidence.py
# Automatic evidence capture for parking violations.
# Saves timestamped screenshots (cropped vehicle + full annotated frame)
# to an evidence directory as proof of the violation.

import cv2
import os
import datetime


class EvidenceCapture:
    """
    Captures and saves photographic evidence when a parking violation is detected.
    """

    def __init__(self, output_dir="evidence"):
        """
        Args:
            output_dir (str): Directory where evidence images will be saved.
        """
        self.output_dir = output_dir
        self._captured_ids = set()  # Track which vehicle IDs already have evidence
        os.makedirs(self.output_dir, exist_ok=True)

    def capture(self, frame, bbox, vehicle_id, vehicle_type, zone_name):
        """
        Save evidence for a violation. Only captures ONCE per vehicle ID.

        Saves two images:
            1. Cropped vehicle image with metadata overlay
            2. Full annotated frame

        Args:
            frame        (np.ndarray): Current annotated frame (full).
            bbox         (tuple):      (x1, y1, x2, y2) bounding box of the vehicle.
            vehicle_id   (int):        Tracker-assigned vehicle ID.
            vehicle_type (str):        Class label (car, truck, bus, motorcycle).
            zone_name    (str):        Zone where violation occurred.

        Returns:
            str | None: Path to the saved full-frame evidence, or None if already captured.
        """
        if vehicle_id in self._captured_ids:
            return None

        self._captured_ids.add(vehicle_id)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"violation_ID{vehicle_id}_{timestamp}"

        # ── Save cropped vehicle image ──────────────────────────────────────
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        # Add padding around the crop
        pad = 20
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        crop = frame[cy1:cy2, cx1:cx2].copy()

        # Add metadata text to the crop
        info_text = f"ID:{vehicle_id} | {vehicle_type.upper()} | {zone_name}"
        time_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(crop, info_text, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(crop, time_text, (5, crop.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        crop_path = os.path.join(self.output_dir, f"{prefix}_crop.jpg")
        cv2.imwrite(crop_path, crop)

        # ── Save full frame snapshot ────────────────────────────────────────
        full_frame = frame.copy()
        # Add a bright border around the violating vehicle
        cv2.rectangle(full_frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 4)

        frame_path = os.path.join(self.output_dir, f"{prefix}_full.jpg")
        cv2.imwrite(frame_path, full_frame)

        print(f"[Evidence] Saved: {crop_path}")
        return frame_path

    def clear(self):
        """Reset the captured set (call on new detection session)."""
        self._captured_ids.clear()

    def get_evidence_dir(self):
        """Return the absolute path to the evidence directory."""
        return os.path.abspath(self.output_dir)
