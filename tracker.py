# tracker.py
# Hybrid Centroid + IoU Tracker for tracking vehicles across video frames.
# Assigns a unique ID to each detected vehicle and maintains that ID
# across frames using a combination of centroid distance and IoU matching.
# Also tracks centroid stability to determine if a vehicle is stationary.

import numpy as np
from collections import OrderedDict, deque


class CentroidTracker:
    """
    Tracks objects (vehicles) across video frames using a hybrid approach:
      1. IoU (Intersection over Union) matching for overlapping bounding boxes.
      2. Centroid distance matching as a fallback.

    Each vehicle gets a unique integer ID that persists across frames.
    """

    def __init__(self, max_disappeared=50, iou_threshold=0.25):
        """
        Args:
            max_disappeared (int): Number of consecutive frames an object can
                                   be missing before it is deregistered.
            iou_threshold (float): Minimum IoU to consider a match between
                                   an existing object and a new detection.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()      # { id: centroid (x, y) }
        self.bboxes = OrderedDict()       # { id: (x1, y1, x2, y2) }
        self.disappeared = OrderedDict()  # { id: frames_missing }
        self.position_history = {}        # { id: deque of (cx, cy) }
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.stability_window = 15       # Number of frames to check for stability
        self.stability_threshold = 25    # Max pixel displacement to count as stationary

    def register(self, centroid, bbox):
        """Register a new object with the next available ID."""
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.position_history[self.next_object_id] = deque(maxlen=self.stability_window)
        self.position_history[self.next_object_id].append(tuple(centroid))
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object that has been missing too long."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        self.position_history.pop(object_id, None)

    @staticmethod
    def _compute_iou(boxA, boxB):
        """
        Compute Intersection over Union between two bounding boxes.

        Args:
            boxA (tuple): (x1, y1, x2, y2)
            boxB (tuple): (x1, y1, x2, y2)

        Returns:
            float: IoU value between 0.0 and 1.0.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        if inter_area == 0:
            return 0.0

        area_A = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        area_B = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = area_A + area_B - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, rects):
        """
        Update tracker with new bounding boxes from the current frame.
        Uses IoU matching first, then falls back to centroid distance.

        Args:
            rects (list): List of bounding boxes as (x1, y1, x2, y2).

        Returns:
            OrderedDict: Mapping of { object_id: centroid (x, y) }.
        """
        # If no detections in this frame, mark all existing objects as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Compute centroids for current detections
        input_centroids = np.array(
            [[(r[0] + r[2]) // 2, (r[1] + r[3]) // 2] for r in rects],
            dtype=int
        )
        input_bboxes = list(rects)

        # If no existing objects, register all new centroids
        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self.register(c, input_bboxes[i])

        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            object_bboxes = [self.bboxes[oid] for oid in object_ids]

            # ── Phase 1: IoU matching ────────────────────────────────────────
            num_existing = len(object_ids)
            num_input = len(input_centroids)

            iou_matrix = np.zeros((num_existing, num_input), dtype=float)
            for i in range(num_existing):
                for j in range(num_input):
                    iou_matrix[i, j] = self._compute_iou(
                        object_bboxes[i], input_bboxes[j]
                    )

            used_rows = set()
            used_cols = set()

            # Greedily match by highest IoU first
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = iou_matrix.max()
                if max_val < self.iou_threshold:
                    break
                max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                row, col = max_idx

                if row in used_rows or col in used_cols:
                    iou_matrix[row, col] = 0
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                if object_id in self.position_history:
                    self.position_history[object_id].append(tuple(input_centroids[col]))
                used_rows.add(row)
                used_cols.add(col)
                iou_matrix[row, :] = 0
                iou_matrix[:, col] = 0

            # ── Phase 2: Centroid fallback for unmatched ─────────────────────
            unmatched_rows = set(range(num_existing)) - used_rows
            unmatched_cols = set(range(num_input)) - used_cols

            if unmatched_rows and unmatched_cols:
                um_rows = sorted(unmatched_rows)
                um_cols = sorted(unmatched_cols)

                um_existing = object_centroids[um_rows]
                um_input = input_centroids[list(um_cols)]

                D = np.linalg.norm(
                    um_existing[:, np.newaxis] - um_input[np.newaxis, :],
                    axis=2,
                )

                row_order = D.min(axis=1).argsort()
                col_order = D.argmin(axis=1)[row_order]

                local_used_rows = set()
                local_used_cols = set()

                for r, c in zip(row_order, col_order):
                    if r in local_used_rows or c in local_used_cols:
                        continue

                    orig_row = um_rows[r]
                    orig_col = um_cols[c] if isinstance(um_cols, list) else sorted(um_cols)[c]

                    object_id = object_ids[orig_row]
                    self.objects[object_id] = input_centroids[orig_col]
                    self.bboxes[object_id] = input_bboxes[orig_col]
                    self.disappeared[object_id] = 0
                    if object_id in self.position_history:
                        self.position_history[object_id].append(tuple(input_centroids[orig_col]))

                    used_rows.add(orig_row)
                    used_cols.add(orig_col)
                    local_used_rows.add(r)
                    local_used_cols.add(c)

            # Handle unmatched existing objects (disappeared)
            for row in set(range(num_existing)) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register brand new detections that had no match
            for col in set(range(num_input)) - used_cols:
                self.register(input_centroids[col], input_bboxes[col])

        return self.objects

    def get_bbox(self, object_id):
        """
        Return the stored bounding box for a tracked object.

        Args:
            object_id (int): Tracker-assigned ID.

        Returns:
            tuple | None: (x1, y1, x2, y2) or None if not found.
        """
        return self.bboxes.get(object_id)

    def is_stable(self, object_id):
        """
        Check if a vehicle's centroid has been stable (not moving much)
        over the last N frames. This is a reliable way to detect if a
        vehicle is truly stationary vs. just passing through.

        Args:
            object_id (int): Tracker-assigned ID.

        Returns:
            bool: True if the vehicle hasn't moved more than the
                  stability_threshold pixels in recent frames.
        """
        history = self.position_history.get(object_id)
        if not history or len(history) < 5:
            return False  # Need at least 5 frames of history

        positions = np.array(list(history))
        # Max displacement from the first recorded position
        first = positions[0]
        displacements = np.linalg.norm(positions - first, axis=1)
        max_displacement = displacements.max()

        return max_displacement < self.stability_threshold
