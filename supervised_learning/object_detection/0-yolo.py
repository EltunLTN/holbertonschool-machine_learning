#!/usr/bin/env python3
"""YOLO object detection module."""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize a Yolo instance."""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [
                line.strip()
                for line in f.readlines()
            ]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
