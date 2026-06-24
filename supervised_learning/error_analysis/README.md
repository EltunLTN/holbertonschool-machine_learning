# Error Analysis

This directory contains error analysis tasks from the Holberton machine learning curriculum. The exercises cover confusion matrices and the most common classification metrics used to evaluate model performance.

## Learning Objectives

- Build a confusion matrix from labels and predictions
- Compute sensitivity, precision, specificity, and F1 score
- Use confusion-matrix metrics to evaluate binary classifiers

## Files

- `0-create_confusion.py`: Defines `create_confusion_matrix(labels, logits)`
- `1-sensitivity.py`: Defines `sensitivity(confusion)`
- `2-precision.py`: Defines `precision(confusion)`
- `3-specificity.py`: Defines `specificity(confusion)`
- `4-f1_score.py`: Defines `f1_score(confusion)`

## Requirements

- Python 3.x
- NumPy
- `pycodestyle` style compliance where required by the checker

## Usage

Run any task file directly with Python, for example:

```bash
python3 0-create_confusion.py
```

Refer to each file's docstring for the expected arguments and return values.
.