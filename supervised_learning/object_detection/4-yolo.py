#!/usr/bin/env python3
"""YOLO object detection module."""

import os

import cv2
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

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            box_confidence = output[..., 4:5]
            box_class_prob = output[..., 5:]

            cx = np.arange(grid_width)
            cy = np.arange(grid_height)

            cx, cy = np.meshgrid(cx, cy)

            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            bx = 1 / (1 + np.exp(-t_x)) + cx
            by = 1 / (1 + np.exp(-t_y)) + cy

            pw = anchor[:, 0]
            ph = anchor[:, 1]

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            bx = bx / grid_width * image_width
            by = by / grid_height * image_height

            bw = bw / grid_width * image_width
            bh = bh / grid_height * image_height

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

    def non_max_suppression(
            self, filtered_boxes, box_classes, box_scores):
        """Perform non-max suppression on filtered boxes."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for class_id in np.unique(box_classes):
            class_mask = box_classes == class_id

            boxes = filtered_boxes[class_mask]
            classes = box_classes[class_mask]
            scores = box_scores[class_mask]

            order = np.argsort(scores)[::-1]

            while order.size > 0:
                index = order[0]

                box_predictions.append(boxes[index])
                predicted_box_classes.append(classes[index])
                predicted_box_scores.append(scores[index])

                if order.size == 1:
                    break

                remaining = order[1:]

                x1 = boxes[index, 0]
                y1 = boxes[index, 1]
                x2 = boxes[index, 2]
                y2 = boxes[index, 3]

                xx1 = np.maximum(x1, boxes[remaining, 0])
                yy1 = np.maximum(y1, boxes[remaining, 1])
                xx2 = np.minimum(x2, boxes[remaining, 2])
                yy2 = np.minimum(y2, boxes[remaining, 3])

                width = np.maximum(0, xx2 - xx1)
                height = np.maximum(0, yy2 - yy1)

                intersection = width * height

                area_index = (x2 - x1) * (y2 - y1)

                area_remaining = (
                    (boxes[remaining, 2] - boxes[remaining, 0]) *
                    (boxes[remaining, 3] - boxes[remaining, 1])
                )

                union = area_index + area_remaining - intersection
                iou = intersection / union

                order = remaining[iou <= self.nms_t]

        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores)
        )

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder."""
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            image = cv2.imread(path)

            if image is not None:
                images.append(image)
                image_paths.append(path)

        return images, image_paths
