# visualizer.py
# Handles all OpenCV drawing operations:
#   - Vehicle bounding boxes (color-coded by status)
#   - Alert banner for active violations
#   - Bottom info panel with statistics
#   - Violation heatmap overlay (NEW)
#   - Stationarity indicator (NEW)

import cv2
import numpy as np


class Visualizer:
    """
    Draws detection results and status overlays onto video frames.

    Color scheme:
        Green    — Normal vehicle (outside any zone)
        Orange   — Vehicle inside a zone, moving (not yet timing)
        Yellow   — Vehicle inside a zone, stationary (timer running)
        Red      — Illegal parking violation (threshold exceeded)

    New features:
        - Heatmap overlay for violation hotspots
        - Motion/stationary status indicator on each vehicle
    """

    COLOR_NORMAL     = (50, 200, 50)    # Green
    COLOR_MOVING     = (30, 150, 255)   # Orange — in zone but moving
    COLOR_TIMING     = (50, 220, 255)   # Yellow — stationary, timer running
    COLOR_VIOLATION  = (30, 30, 220)    # Red    — violation
    FONT             = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.heatmap_accumulator = None
        self.show_heatmap = False

    # ------------------------------------------------------------------ #
    #  Heatmap                                                             #
    # ------------------------------------------------------------------ #

    def init_heatmap(self, width, height):
        """Initialize or reset the heatmap accumulator."""
        self.heatmap_accumulator = np.zeros((height, width), dtype=np.float32)

    def update_heatmap(self, bbox):
        """Add heat at a violation bounding box location."""
        if self.heatmap_accumulator is None:
            return
        x1, y1, x2, y2 = bbox
        h, w = self.heatmap_accumulator.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        self.heatmap_accumulator[y1:y2, x1:x2] += 1.0

    def draw_heatmap(self, frame):
        """
        Overlay the violation heatmap on the frame.

        Uses COLORMAP_JET for a colorful heat visualization.
        Only drawn when show_heatmap is True.

        Args:
            frame (np.ndarray): Frame to overlay on.

        Returns:
            np.ndarray: Frame with heatmap overlay.
        """
        if not self.show_heatmap or self.heatmap_accumulator is None:
            return frame

        if self.heatmap_accumulator.max() == 0:
            return frame

        # Normalize the heatmap to 0-255
        heat_norm = self.heatmap_accumulator / self.heatmap_accumulator.max()
        heat_norm = (heat_norm * 255).astype(np.uint8)

        # Apply Gaussian blur for smooth heat spread
        heat_norm = cv2.GaussianBlur(heat_norm, (51, 51), 0)

        # Apply colormap
        heat_colored = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        # Only overlay where there is heat (threshold at 30)
        mask = heat_norm > 30
        mask_3c = np.stack([mask] * 3, axis=-1)

        blended = frame.copy()
        blended[mask_3c] = cv2.addWeighted(
            heat_colored, 0.4, frame, 0.6, 0
        )[mask_3c]

        # Add label
        cv2.putText(blended, "VIOLATION HEATMAP", (10, frame.shape[0] - 50),
                    self.FONT, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

        return blended

    # ------------------------------------------------------------------ #
    #  Per-vehicle annotation                                              #
    # ------------------------------------------------------------------ #

    def draw_vehicle(self, frame, bbox, vehicle_id, vehicle_type,
                     duration, is_violation, in_zone, is_stationary=False):
        """
        Draw a bounding box around a vehicle with a status label.

        Args:
            frame         (np.ndarray): Frame to annotate.
            bbox          (tuple):      (x1, y1, x2, y2) bounding box.
            vehicle_id    (int):        Tracker-assigned ID.
            vehicle_type  (str):        Class label.
            duration      (float):      Seconds in zone (0 if not in zone).
            is_violation  (bool):       True if threshold exceeded.
            in_zone       (bool):       True if currently inside a no-parking zone.
            is_stationary (bool):       True if BGS confirms vehicle is stationary.

        Returns:
            np.ndarray: Annotated frame.
        """
        x1, y1, x2, y2 = bbox

        if is_violation:
            color = self.COLOR_VIOLATION
            status = f"!! ILLEGAL PARKING ({duration:.0f}s)"
        elif in_zone and is_stationary:
            color = self.COLOR_TIMING
            status = f"STATIONARY: {duration:.0f}s"
        elif in_zone:
            color = self.COLOR_MOVING
            status = f"In Zone (moving)"
        else:
            color = self.COLOR_NORMAL
            status = vehicle_type.upper()

        label = f"ID:{vehicle_id}  {status}"
        self._draw_box_with_label(frame, x1, y1, x2, y2, label, color)

        # Add a semi-transparent red flash overlay for violations
        if is_violation:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_VIOLATION, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Stationarity indicator dot
        if in_zone:
            dot_color = (0, 255, 255) if is_stationary else (0, 165, 255)
            dot_label = "S" if is_stationary else "M"
            cv2.circle(frame, (x2 - 10, y1 + 10), 8, dot_color, -1)
            cv2.putText(frame, dot_label, (x2 - 14, y1 + 14),
                        self.FONT, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def _draw_box_with_label(self, frame, x1, y1, x2, y2, label, color):
        """Internal helper: draws a rectangle and a filled label background."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Compute label background size
        (text_w, text_h), baseline = cv2.getTextSize(label, self.FONT, 0.48, 1)
        label_y1 = max(y1 - text_h - 8, 0)
        label_y2 = y1

        cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 6, label_y2), color, -1)
        cv2.putText(frame, label, (x1 + 3, label_y2 - 3),
                    self.FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  HUD overlays                                                        #
    # ------------------------------------------------------------------ #

    def draw_alert_banner(self, frame, active_violation_count):
        """
        Draw a red alert banner at the top of the frame when violations exist.

        Args:
            frame                 (np.ndarray): Frame to draw on.
            active_violation_count (int):       Number of current violations.

        Returns:
            np.ndarray: Annotated frame.
        """
        if active_violation_count > 0:
            # Pulsing effect using transparency variation
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), (20, 20, 180), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            text = f"  ALERT: {active_violation_count} ILLEGAL PARKING VIOLATION(S) DETECTED"
            cv2.putText(frame, text, (8, 34),
                        self.FONT, 0.80, (255, 255, 255), 2, cv2.LINE_AA)

            # Warning icon triangles
            pts = np.array([[frame.shape[1] - 50, 8],
                            [frame.shape[1] - 30, 40],
                            [frame.shape[1] - 70, 40]], np.int32)
            cv2.fillPoly(frame, [pts], (0, 200, 255))
            cv2.putText(frame, "!", (frame.shape[1] - 45, 36),
                        self.FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        return frame

    def draw_info_panel(self, frame, total_detected, total_violations, fps,
                        night_mode=False, stationary_count=0):
        """
        Draw a statistics bar at the bottom of the frame.

        Args:
            frame             (np.ndarray): Frame to draw on.
            total_detected    (int):        Vehicles currently tracked.
            total_violations  (int):        Total violations logged so far.
            fps               (float):      Current processing frames per second.
            night_mode        (bool):       Whether CLAHE preprocessing is active.
            stationary_count  (int):        Vehicles confirmed stationary.

        Returns:
            np.ndarray: Annotated frame.
        """
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 36), (w, h), (20, 20, 30), -1)

        night_tag = " | NIGHT MODE" if night_mode else ""
        info = (f"  Vehicles: {total_detected}   |   "
                f"Stationary: {stationary_count}   |   "
                f"Violations: {total_violations}   |   "
                f"FPS: {fps:.1f}{night_tag}")
        cv2.putText(frame, info, (8, h - 10),
                    self.FONT, 0.52, (200, 200, 210), 1, cv2.LINE_AA)
        return frame
