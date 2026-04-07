# background_subtractor.py
# Background subtraction using MOG2 with morphological post-processing.
# Used to confirm whether a detected vehicle is truly STATIONARY before
# the parking timer begins — this eliminates false alerts from temporary
# stops and reduces shadow-triggered false positives.

import cv2
import numpy as np


class BackgroundSubtractorModule:
    """
    Wraps OpenCV's MOG2 background subtractor and adds morphological
    post-processing to produce a clean foreground mask.
    """

    def __init__(self, history=500, var_threshold=50, detect_shadows=True,
                 morph_kernel_size=5, stationary_threshold=0.15):
        """
        Args:
            history (int): Number of frames used to build the background model.
            var_threshold (int): Variance threshold for pixel classification.
            detect_shadows (bool): If True, shadows are detected and marked grey.
            morph_kernel_size (int): Size of the kernel for morphological ops.
            stationary_threshold (float): If the ratio of foreground pixels within
                a bounding box is BELOW this value, the vehicle is considered stationary.
                (A moving vehicle has lots of foreground activity; a parked one has little.)
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.stationary_threshold = stationary_threshold
        self.mask = None

    def apply(self, frame):
        """
        Compute the foreground mask for the current frame.

        Steps:
            1. Apply MOG2 background subtractor
            2. Threshold to binary (removes shadow grey values)
            3. Erode to remove small noise / shadow remnants
            4. Dilate to restore object shapes

        Args:
            frame (np.ndarray): BGR input frame.

        Returns:
            np.ndarray: Binary foreground mask (255 = foreground, 0 = background).
        """
        # Step 1: Background subtraction
        raw_mask = self.bg_subtractor.apply(frame)

        # Step 2: Threshold — remove shadow pixels (grey=127 in MOG2)
        _, binary_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

        # Step 3: Morphological erosion — remove small noise blobs & shadow edges
        eroded = cv2.morphologyEx(binary_mask, cv2.MORPH_ERODE, self.morph_kernel, iterations=1)

        # Step 4: Morphological dilation — restore vehicle boundaries
        dilated = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, self.morph_kernel, iterations=2)

        self.mask = dilated
        return dilated

    def is_stationary(self, bbox):
        """
        Check whether the region inside a bounding box is mostly background
        (i.e. the vehicle is NOT moving / has been absorbed into the background).

        A vehicle that has been parked for a while will have LOW foreground
        activity because MOG2 absorbs static objects into the background model.
        A vehicle that just stopped or is still moving will have HIGH foreground.

        Args:
            bbox (tuple): (x1, y1, x2, y2) bounding box of the vehicle.

        Returns:
            bool: True if the vehicle appears stationary (low motion).
        """
        if self.mask is None:
            return False

        x1, y1, x2, y2 = bbox
        h, w = self.mask.shape[:2]

        # Clamp coordinates to frame boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return False

        roi = self.mask[y1:y2, x1:x2]
        total_pixels = roi.size
        if total_pixels == 0:
            return False

        # Ratio of foreground pixels in the bounding box
        foreground_ratio = np.count_nonzero(roi) / total_pixels

        # Low foreground ratio means vehicle is stationary (absorbed into BG)
        # High foreground ratio means vehicle is moving or just arrived
        return foreground_ratio < self.stationary_threshold

    def get_motion_ratio(self, bbox):
        """
        Return the foreground pixel ratio for a bounding box (for display purposes).

        Args:
            bbox (tuple): (x1, y1, x2, y2).

        Returns:
            float: Ratio of foreground pixels (0.0 to 1.0).
        """
        if self.mask is None:
            return 0.0

        x1, y1, x2, y2 = bbox
        h, w = self.mask.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = self.mask[y1:y2, x1:x2]
        total = roi.size
        return np.count_nonzero(roi) / total if total > 0 else 0.0
