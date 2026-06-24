# Decision Tree

This directory contains decision tree, random forest, and isolation forest tasks from the Holberton machine learning curriculum. The exercises focus on implementing tree-based learning algorithms from scratch and using them for classification and anomaly detection.

## Learning Objectives

- Build binary tree structures with node and leaf classes
- Implement decision tree training and prediction logic
- Train random forest ensembles from multiple decision trees
- Build isolation trees and isolation forests for anomaly detection

## Files

- `0-build_decision_tree.py`: Defines the base `Node`, `Leaf`, and `Decision_Tree` classes
- `1-build_decision_tree.py`: Extends the decision tree implementation with training support
- `2-build_decision_tree.py`: Adds improved tree construction logic
- `3-build_decision_tree.py`: Adds additional tree-building behavior and traversal helpers
- `4-build_decision_tree.py`: Extends the decision tree implementation with more robust splitting
- `5-build_decision_tree.py`: Improves tree training and prediction behavior
- `6-build_decision_tree.py`: Adds more advanced decision tree construction
- `7-build_decision_tree.py`: Defines the `Random_Forest` class
- `8-build_decision_tree.py`: Finalizes decision tree functionality used by later tasks
- `9-random_forest.py`: Defines the `Random_Forest` ensemble class
- `10-isolation_tree.py`: Defines the `Isolation_Random_Tree` class
- `11-isolation_forest.py`: Defines the `Isolation_Random_Forest` class

## Requirements

- Python 3.x
- NumPy
- `pycodestyle` style compliance where required by the checker

## Usage

Run any task file directly with Python, for example:

```bash
python3 0-build_decision_tree.py
```

Refer to each file's docstring for the expected arguments and return values.
.