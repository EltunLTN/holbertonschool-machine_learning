# Math - Plotting

This directory contains Matplotlib visualization tasks from the Holberton machine learning curriculum. Scripts generate line graphs, scatter plots, histograms, stacked bar charts, and combined multi-panel figures.

## Learning Objectives

- Create basic plots with NumPy and Matplotlib
- Customize axes, labels, titles, legends, and scales
- Visualize simulated real-world data (height/weight, radioactive decay, grades)
- Combine multiple plot types in a single figure
- Apply colormaps and 3D scatter plots for advanced visualization

## Files

- `0-line.py`: Defines `line()` — plots y = x³ as a solid red line
- `1-scatter.py`: Defines `scatter()` — men's height vs weight scatter plot
- `2-change_scale.py`: Defines `change_scale()` — C-14 exponential decay with log y-axis
- `3-two.py`: Defines `two()` — C-14 and Ra-226 decay on the same axes
- `4-frequency.py`: Defines `frequency()` — histogram of student project grades
- `5-all_in_one.py`: Defines `all_in_one()` — all five previous plots in one 3×2 figure
- `6-bars.py`: Defines `bars()` — stacked bar graph of fruit per person
- `100-gradient.py`: Defines `gradient()` — mountain elevation scatter with colorbar
- `101-pca.py`: 3D PCA visualization of the Iris dataset (requires `pca.npz`)

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- `pycodestyle` style compliance (where required by checker)

## Usage

Run a plotting script:

```bash
python3 0-line.py
```

Each script calls its plotting function and displays the figure with `plt.show()`.
