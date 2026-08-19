#!/usr/bin/env python3
"""YOLO object detection module."""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """YOLO class."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo class."""
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

    def process_outputs(self, outputs, image_size):
        """Process the outputs of the Darknet model."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for output, anchor in zip(outputs, self.anchors):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract raw predictions
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            box_confidence = output[..., 4:5]
            box_class_prob = output[..., 5:]

            # Create grid coordinates
            cx = np.arange(grid_width)
            cy = np.arange(grid_height)

            cx, cy = np.meshgrid(cx, cy)

            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            # Apply sigmoid to x and y
            bx = 1 / (1 + np.exp(-t_x)) + cx
            by = 1 / (1 + np.exp(-t_y)) + cy

            # Anchor dimensions
            pw = anchor[:, 0]
            ph = anchor[:, 1]

            # Calculate width and height
            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            # Convert coordinates to original image size
            bx = bx / grid_width * image_width
            by = by / grid_height * image_height

            bw = bw / grid_width * image_width
            bh = bh / grid_height * image_height

            # Convert center coordinates to corner coordinates
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = bx + bw / 2
            y2 = by + bh / 2

            box = np.stack((x1, y1, x2, y2), axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
