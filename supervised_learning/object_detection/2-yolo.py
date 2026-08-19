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
        """Process outputs of the Darknet model."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size

        for output, anchor in zip(outputs, self.anchors):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            box_confidence = output[..., 4:5]
            box_class_prob = output[..., 5:]

            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx, cy = np.meshgrid(cx, cy)

            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            bx = 1 / (1 + np.exp(-t_xy[..., 0])) + cx
            by = 1 / (1 + np.exp(-t_xy[..., 1])) + cy

            pw = anchor[:, 0]
            ph = anchor[:, 1]

            bw = pw * np.exp(t_wh[..., 0])
            bh = ph * np.exp(t_wh[..., 1])

            bx = bx / grid_w * image_w
            by = by / grid_h * image_h

            bw = bw / grid_w * image_w
            bh = bh / grid_h * image_h

            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = bx + bw / 2
            y2 = by + bh / 2

            box = np.stack((x1, y1, x2, y2), axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes based on their confidence scores."""
        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):

            scores = confidence * class_probs

            classes = np.argmax(scores, axis=-1)
            scores = np.max(scores, axis=-1)

            mask = scores >= self.class_t

            filtered_boxes.append(box[mask])
            filtered_classes.append(classes[mask])
            filtered_scores.append(scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        filtered_classes = np.concatenate(filtered_classes, axis=0)
        filtered_scores = np.concatenate(filtered_scores, axis=0)

        return filtered_boxes, filtered_classes, filtered_scores
