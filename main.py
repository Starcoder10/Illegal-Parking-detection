# main.py
# Entry point for the Illegal Parking Detection System (v2 — Upgraded).
# Launches the Tkinter dashboard. The core detection loop (run_detection)
# is also defined here and called from the dashboard in a background thread.
#
# v2 Upgrades:
#   - CLAHE preprocessing for low-light enhancement
#   - Background subtraction (MOG2) for stationarity confirmation
#   - Evidence capture for violations
#   - Heatmap integration
#   - IoU-enhanced tracking

import cv2
import time
import datetime
import os

from detector              import VehicleDetector
from tracker               import CentroidTracker
from roi                   import ROIManager
from timer_check           import ParkingTimer
from visualizer            import Visualizer
from preprocessor          import Preprocessor
from background_subtractor import BackgroundSubtractorModule
from evidence              import EvidenceCapture


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: extract the first frame of a video (used by dashboard for preview)
# ──────────────────────────────────────────────────────────────────────────────

def get_first_frame(video_path):
    """
    Open a video file and return its very first frame as a BGR NumPy array.

    Args:
        video_path (str): Path to the video file.

    Returns:
        np.ndarray | None: First frame, or None if the file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ──────────────────────────────────────────────────────────────────────────────
#  Core detection loop (v2 — Upgraded)
# ──────────────────────────────────────────────────────────────────────────────

def run_detection(
    video_path,
    zones,
    threshold,
    output_path=None,
    frame_callback=None,
    log_callback=None,
    stop_flag=None,
    stats_callback=None,
    night_mode=False,
    confidence_threshold=0.40,
    show_heatmap=False,
):
    """
    Process a video file frame-by-frame to detect illegal parking.

    Args:
        video_path           (str):       Path to the input video.
        zones                (list):      Polygon zones from ROIManager.
        threshold            (int):       Dwell-time limit in seconds.
        output_path          (str|None):  Where to save the annotated output video.
        frame_callback       (callable):  Called with each annotated frame (for live display).
        log_callback         (callable):  Called with a violation dict when a new violation fires.
        stop_flag            (list):      Single-element list [False]; set to [True] to stop.
        stats_callback       (callable):  Called with a stats dict every frame.
        night_mode           (bool):      Enable CLAHE preprocessing for low-light.
        confidence_threshold (float):     YOLO detection confidence threshold.
        show_heatmap         (bool):      Enable violation heatmap overlay.

    Returns:
        list: All recorded violation dicts.
    """
    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Main] ERROR: Cannot open video '{video_path}'")
        return []

    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Optional output writer ───────────────────────────────────────────────
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))
        print(f"[Main] Output will be saved to: {output_path}")

    # ── Initialise all modules ───────────────────────────────────────────────
    detector      = VehicleDetector(confidence_threshold=confidence_threshold)
    tracker       = CentroidTracker(max_disappeared=40, iou_threshold=0.25)
    roi_manager   = ROIManager()
    parking_timer = ParkingTimer(threshold_seconds=threshold)
    visualizer    = Visualizer()
    preprocessor  = Preprocessor()
    bg_subtractor = BackgroundSubtractorModule()
    evidence      = EvidenceCapture()

    # Configure modules
    preprocessor.set_night_mode(night_mode)
    visualizer.init_heatmap(width, height)
    visualizer.show_heatmap = show_heatmap

    roi_manager.zones       = zones
    roi_manager._zone_names = [f"Zone {i + 1}" for i in range(len(zones))]

    # ── Per-ID metadata stores ───────────────────────────────────────────────
    vehicle_type_map = {}   # { vehicle_id: class_name }
    vehicle_bbox_map = {}   # { vehicle_id: (x1,y1,x2,y2) }

    frame_count  = 0
    start_time   = time.time()
    fps_display  = 0.0

    print("[Main] Detection started (v2 — Upgraded).")
    print(f"[Main] Night mode: {'ON' if night_mode else 'OFF'} | "
          f"Confidence: {confidence_threshold} | "
          f"Threshold: {threshold}s | "
          f"Heatmap: {'ON' if show_heatmap else 'OFF'}")

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        # Honour stop signal from dashboard
        if stop_flag and stop_flag[0]:
            print("[Main] Stop signal received.")
            break

        ret, frame = cap.read()
        if not ret:
            print("[Main] End of video.")
            break

        frame_count += 1

        # Recalculate display FPS every 15 frames
        if frame_count % 15 == 0:
            elapsed     = time.time() - start_time
            fps_display = frame_count / elapsed if elapsed > 0 else 0.0

        # ── Step 1: Preprocess frame (CLAHE if night mode) ───────────────────
        processed_frame = preprocessor.process(frame)

        # ── Step 2: Background subtraction ───────────────────────────────────
        bg_subtractor.apply(processed_frame)

        # ── Step 3: Detect vehicles on preprocessed frame ────────────────────
        detections = detector.detect(processed_frame)
        rects      = [(d[0], d[1], d[2], d[3]) for d in detections]

        # ── Step 4: Update tracker (IoU + centroid) ──────────────────────────
        objects = tracker.update(rects)

        # ── Step 5: Map tracker IDs to detection metadata ────────────────────
        for vid, centroid in objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            for det in detections:
                x1, y1, x2, y2, vtype, _conf = det
                dcx = (x1 + x2) // 2
                dcy = (y1 + y2) // 2
                if abs(dcx - cx) < 40 and abs(dcy - cy) < 40:
                    vehicle_type_map[vid] = vtype
                    vehicle_bbox_map[vid] = (x1, y1, x2, y2)
                    break

        # ── Step 6: Draw no-parking zones ────────────────────────────────────
        display_frame = processed_frame.copy()
        roi_manager.draw_zones(display_frame)

        active_violations  = 0
        stationary_count   = 0

        # ── Step 7: Per-vehicle zone check, stationarity & timer logic ───────
        for vid, centroid in objects.items():
            cx, cy    = int(centroid[0]), int(centroid[1])
            zone_idx  = roi_manager.get_vehicle_zone((cx, cy))
            vtype     = vehicle_type_map.get(vid, "vehicle")
            bbox      = vehicle_bbox_map.get(vid)

            in_zone       = zone_idx >= 0
            is_violation  = False
            is_stationary = False
            duration      = 0.0

            # Check stationarity using centroid stability (much more reliable than BGS)
            is_stationary = tracker.is_stable(vid)

            if is_stationary:
                stationary_count += 1

            if in_zone:
                zone_name = roi_manager._zone_names[zone_idx]

                # Update stationarity status in timer
                parking_timer.set_stationary(vid, is_stationary)

                # Timer only begins if vehicle is stationary (gated)
                parking_timer.vehicle_in_zone(vid, zone_idx, zone_name)
                duration     = parking_timer.get_duration(vid)
                is_violation = parking_timer.is_violation(vid)

                if is_violation:
                    active_violations += 1
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    new_entry = parking_timer.record_violation(vid, zone_name, vtype, ts)

                    # Update heatmap
                    if bbox:
                        visualizer.update_heatmap(bbox)

                    # Capture evidence (only on first violation per vehicle)
                    if new_entry and bbox:
                        evidence.capture(display_frame, bbox, vid, vtype, zone_name)

                    # Fire log callback only for brand-new violations
                    if new_entry and log_callback:
                        log_callback(parking_timer.violations[vid].copy())
            else:
                parking_timer.vehicle_out_of_zone(vid)

            # ── Step 8: Annotate the frame ───────────────────────────────────
            if bbox:
                visualizer.draw_vehicle(
                    display_frame, bbox, vid, vtype, duration,
                    is_violation, in_zone, is_stationary
                )

        # ── Step 9: Heatmap overlay ──────────────────────────────────────────
        display_frame = visualizer.draw_heatmap(display_frame)

        # ── Step 10: HUD overlays ────────────────────────────────────────────
        total_violations = len(parking_timer.violations)
        visualizer.draw_alert_banner(display_frame, active_violations)
        visualizer.draw_info_panel(
            display_frame, len(objects), total_violations, fps_display,
            night_mode=preprocessor.night_mode,
            stationary_count=stationary_count,
        )

        # ── Callbacks ────────────────────────────────────────────────────────
        if stats_callback:
            stats_callback({
                "total_detected":   len(objects),
                "total_violations": total_violations,
                "active_zones":     len(zones),
                "stationary":       stationary_count,
                "zone_stats":       parking_timer.get_zone_stats(),
                "timeline":         parking_timer.get_timeline(),
            })

        if frame_callback:
            frame_callback(display_frame)

        if writer:
            writer.write(display_frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"[Main] Output video saved: {output_path}")

    print(f"[Main] Detection complete. {len(parking_timer.violations)} violation(s) recorded.")
    return parking_timer.get_all_violations()


# ──────────────────────────────────────────────────────────────────────────────
#  Launch the dashboard
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dashboard import launch_dashboard
    launch_dashboard()
