# Object Detection - YOLO

This directory contains implementation of the YOLO (You Only Look Once) object detection algorithm from the Holberton machine learning curriculum. YOLO is a real-time object detection system that predicts bounding boxes and class probabilities in a single forward pass through the network.

## Learning Objectives

- Understand the YOLO architecture and how it frames detection as a regression problem
- Implement anchor boxes and multi-scale predictions
- Compute Intersection over Union (IoU) for bounding box evaluation
- Perform Non-Maximum Suppression (NMS) for duplicate detection removal
- Load pre-trained YOLO models and perform predictions

## Files

- `0-yolo.py`: Defines `Yolo()` class — basic YOLO model initialization
- `1-yolo.py`: Adds `process_outputs(outputs, image_size)` — process raw network predictions
- `2-yolo.py`: Adds `filter_boxes(boxes, box_confidences, box_class_probs, class_t)` — filter predictions by confidence
- `3-yolo.py`: Adds `iou(box1, box2)` — compute Intersection over Union between boxes
- `4-yolo.py`: Adds `non_max_suppression(filtered_boxes, box_classes, box_scores)` — remove duplicate detections
- `5-yolo.py`: Adds `load_images(folder_path)` — load and preprocess images
- `6-yolo.py`: Adds `predict(net, folder_path, prob_t=0.5, iou_t=0.45)` — full detection pipeline
- `7-yolo.py`: Adds visualization and output formatting methods

## Requirements

- Python 3.x
- TensorFlow/Keras 2.x or higher
- NumPy
- OpenCV (cv2) for image processing
- `pycodestyle` style compliance where required
- Pre-trained YOLO weights (typically YOLOv3)

## Key Concepts

**Grid-based Detection**: Image is divided into SxS grid; each cell predicts bounding boxes and class probabilities

**Anchor Boxes**: Pre-defined box shapes to help network learn diverse object sizes

**Bounding Box Regression**: Network predicts (x, y, width, height) offsets relative to anchor boxes

**Objectness Score**: Confidence that a cell contains an object (IoU between prediction and ground truth)

**Intersection over Union (IoU)**: Measures overlap between predicted and ground truth boxes: IoU = Intersection / Union

**Non-Maximum Suppression**: Removes duplicate/overlapping predictions by keeping boxes with highest confidence

**Multi-scale Detection**: YOLO makes predictions at multiple scales to detect objects of different sizes

## Usage

```python
from yolo import Yolo

# Initialize YOLO model
yolo = Yolo('yolov3.cfg', 'yolov3.weights', 'coco.names', 0.6)

# Detect objects in images
detections = yolo.predict('images_folder/', prob_t=0.5, iou_t=0.45)

# Access detections
for image, boxes, classes, scores in detections:
    print(f"Found {len(boxes)} objects in {image}")
    for box, class_id, score in zip(boxes, classes, scores):
        print(f"  Class: {class_id}, Confidence: {score:.2f}, Box: {box}")
```

## YOLO Advantages

- Real-time inference speed (>30 FPS)
- Detects multiple objects in single pass
- Learns generalizable object representations
- Reason globally about image context

## References

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection
