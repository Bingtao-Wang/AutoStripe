#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualization module for ORB-SLAM3 testing."""

import cv2
import numpy as np


class SLAMVisualizer:
    """Real-time SLAM visualization (stereo ORB features only)."""

    def __init__(self, width=752, height=480):
        self.width = width
        self.height = height
        self.enabled = False
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)

    def update(self, stereo_left, stereo_right, ate=None):
        """Update stereo visualization with ORB keypoints."""
        if not self.enabled:
            return

        gray_left = cv2.cvtColor(stereo_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(stereo_right, cv2.COLOR_BGR2GRAY)
        kp_left = self.orb.detect(gray_left, None)
        kp_right = self.orb.detect(gray_right, None)
        img_left = cv2.drawKeypoints(gray_left, kp_left, None, color=(0, 255, 0), flags=0)
        img_right = cv2.drawKeypoints(gray_right, kp_right, None, color=(0, 255, 0), flags=0)

        stereo_display = np.hstack([img_left, img_right])

        cv2.putText(stereo_display, "Left Camera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(stereo_display, "Right Camera", (self.width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if ate is not None:
            cv2.putText(stereo_display, "ATE: {:.3f}m".format(ate), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Stereo Camera", stereo_display)

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            cv2.destroyAllWindows()
