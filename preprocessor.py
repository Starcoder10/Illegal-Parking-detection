# preprocessor.py
# Frame preprocessing pipeline using Digital Image Processing techniques.
# Provides CLAHE (Contrast Limited Adaptive Histogram Equalization) for
# low-light / night-time enhancement and optional Gaussian denoising.

import cv2
import numpy as np


class Preprocessor:
    """
    Applies image enhancement techniques to video frames before detection.
    Designed to improve YOLOv8 accuracy in poor lighting conditions.
    """

    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit (float): CLAHE contrast limit. Higher = more enhancement.
            tile_grid_size (tuple): Grid size for CLAHE. Smaller = more local contrast.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        self.night_mode = False

    def set_night_mode(self, enabled):
        """Toggle CLAHE enhancement on/off."""
        self.night_mode = enabled

    def set_clip_limit(self, clip_limit):
        """Update the CLAHE clip limit and recreate the CLAHE object."""
        self.clip_limit = clip_limit
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

    def apply_clahe(self, frame):
        """
        Enhance frame contrast using CLAHE on the L channel of LAB color space.

        Steps:
            1. Convert BGR → LAB
            2. Apply CLAHE to the L (lightness) channel
            3. Convert LAB → BGR

        Args:
            frame (np.ndarray): BGR input frame.

        Returns:
            np.ndarray: Contrast-enhanced BGR frame.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to lightness channel
        l_enhanced = self.clahe.apply(l_channel)

        # Merge channels back and convert to BGR
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def apply_denoise(self, frame, kernel_size=3):
        """
        Apply Gaussian blur to reduce noise.

        Args:
            frame (np.ndarray): Input frame.
            kernel_size (int): Gaussian kernel size (must be odd).

        Returns:
            np.ndarray: Denoised frame.
        """
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def process(self, frame):
        """
        Full preprocessing pipeline. Only applied when night_mode is ON.

        Args:
            frame (np.ndarray): Raw BGR frame.

        Returns:
            np.ndarray: Processed frame (enhanced if night_mode, else original).
        """
        if not self.night_mode:
            return frame

        # Step 1: CLAHE contrast enhancement
        enhanced = self.apply_clahe(frame)

        # Step 2: Light denoising to reduce artifacts from enhancement
        enhanced = self.apply_denoise(enhanced, kernel_size=3)

        return enhanced
