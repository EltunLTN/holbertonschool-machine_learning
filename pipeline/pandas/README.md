# Pipeline - Pandas

This directory contains Pandas data pipeline tasks from the Holberton machine learning curriculum. Tasks progress from creating DataFrames to loading CSV data, cleaning missing values, and visualizing time series.

## Learning Objectives

- Create DataFrames from NumPy arrays and Python dictionaries
- Load, rename, slice, and index tabular data from files
- Handle missing values with forward fill and drop operations
- Concatenate DataFrames and build hierarchical indexes
- Compute descriptive statistics and plot financial time series

## Files

- `0-from_numpy.py`: Defines `from_numpy(array)` — DataFrame from array with alphabetical column labels
- `1-from_dictionary.py`: Creates a DataFrame from a dictionary
- `2-from_file.py`: Defines `from_file(filename, delimiter)` — loads CSV data into a DataFrame
- `3-rename.py`: Defines `rename(df)` — renames and converts Timestamp column
- `4-array.py`: Defines `array(df)` — converts selected columns to a NumPy array
- `5-slice.py`: Defines `slice(df)` — slices rows by date range
- `6-flip_switch.py`: Defines `flip_switch(df)` — transposes the DataFrame
- `7-high.py`: Defines `high(df)` — sorts rows by High price descending
- `8-prune.py`: Defines `prune(df)` — drops rows with NaN in Open
- `9-fill.py`: Defines `fill(df)` — forward-fills missing Close/High/Low values
- `10-index.py`: Defines `index(df)` — sets Timestamp as the index
- `11-concat.py`: Defines `concat(df1, df2)` — concatenates two DataFrames
- `12-hierarchy.py`: Defines `hierarchy(df1, df2)` — hierarchical concatenation with MultiIndex
- `13-analyze.py`: Defines `analyze(df)` — descriptive statistics excluding Timestamp
- `14-visualize.py`: Plots normalized Coinbase BTC/USD OHLC data (requires CSV file)

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib (for `14-visualize.py`)
- `pycodestyle` style compliance (where required by checker)

## Usage

Run Python task files with:

```bash
python3 0-from_numpy.py
```

For file-based tasks, ensure the CSV dataset is in the working directory:

```bash
python3 14-visualize.py
```
