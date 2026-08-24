# Pipeline - Data Processing and Preparation

This directory contains data pipeline techniques and tools from the Holberton machine learning curriculum. It covers essential skills for loading, preprocessing, transforming, and preparing data for machine learning models. Effective data pipelines are critical for model performance and reproducibility.

## Learning Objectives

- Master data loading and preprocessing with Pandas and NumPy
- Perform exploratory data analysis (EDA) on datasets
- Handle missing values, outliers, and data quality issues
- Transform and engineer features for machine learning
- Apply data augmentation to increase training data
- Build efficient, reproducible data processing workflows
- Understand best practices for data pipeline design

## Subdirectories

### 1. **pandas/** - Data Manipulation with Pandas
   - Loading data from various formats (CSV, JSON, SQL databases)
   - DataFrame exploration and basic statistics
   - Data cleaning and preprocessing
   - Feature selection and engineering
   - Data merging, grouping, and aggregation
   - Handling categorical and numerical variables

### 2. **data_augmentation/** - Expanding Training Data
   - Image transformation techniques (flip, rotate, crop)
   - Color space modifications (brightness, contrast, hue)
   - Geometric augmentation for robustness
   - Synthetic data generation
   - PCA-based augmentation methods

## Requirements

- Python 3.x
- Pandas (primary tool for tabular data manipulation)
- NumPy (numerical operations)
- TensorFlow/Keras (for data pipelines and augmentation)
- Scikit-learn (preprocessing utilities and transformers)
- Matplotlib/Seaborn (data visualization)
- OpenCV (image processing for augmentation)
- SQLAlchemy (optional, for SQL databases)
- `pycodestyle` style compliance where required

## Key Concepts

### Data Pipeline Stages

```
Raw Data → Load → Clean → Transform → Feature Engineering → Model Training
    ↑                                                           ↓
    └─────────── Monitoring & Feedback ←─────────────────────┘
```

### Data Cleaning
- **Missing Values**: Drop, mean/median imputation, KNN imputation
- **Outliers**: Detection and handling (IQR, Z-score, isolation forests)
- **Duplicates**: Identification and removal
- **Inconsistencies**: Fixing format issues, standardizing units

### Data Transformation
- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Encoding**: One-hot encoding, label encoding, target encoding
- **Discretization**: Binning continuous variables
- **Handling Skewness**: Log transformation, Box-Cox transformation

### Feature Engineering
- **Feature Extraction**: Creating new features from raw data
- **Feature Selection**: Identifying most important features
- **Dimensionality Reduction**: PCA, t-SNE, feature selection
- **Feature Interactions**: Polynomial features, cross-features

### Data Augmentation
- **Geometric**: Rotation, flip, crop, affine transformations
- **Color**: Brightness, contrast, hue, saturation adjustments
- **Noise**: Adding random noise for robustness
- **Mixing**: Mixup, CutMix for synthetic examples

## Typical Pipeline Workflow

```python
import pandas as pd
from pipeline.pandas_tools import load_data, clean_data, engineer_features
from pipeline.augmentation import augment_images

# Step 1: Load Data
df = load_data('data.csv')
print(df.head())
print(df.info())

# Step 2: Clean Data
df = clean_data(df)  # Handle missing values, duplicates, outliers

# Step 3: Engineer Features
df = engineer_features(df)  # Create new features

# Step 4: Split Data
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# Step 5: Transform Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Augment Training Data (for images)
X_train_augmented = augment_images(X_train_scaled)

# Step 7: Train Model
model = train_model(X_train_augmented, y_train)
```

## Pandas Module Concepts

### DataFrame Basics
```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Explore data
print(df.head())          # First 5 rows
print(df.info())          # Data types and missing values
print(df.describe())      # Statistical summary

# Access data
df['column_name']         # Single column
df[['col1', 'col2']]      # Multiple columns
df.loc[0]                 # Row by label
df.iloc[0]                # Row by position
```

### Data Cleaning Examples
```python
# Handle missing values
df.fillna(method='ffill')          # Forward fill
df.fillna(df.mean())               # Fill with mean
df.dropna()                        # Drop missing rows

# Remove duplicates
df.drop_duplicates()

# Filter outliers
df = df[(df['column'] >= lower) & (df['column'] <= upper)]

# Fix data types
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
```

### Feature Engineering
```python
# Create new features
df['feature1_squared'] = df['feature1'] ** 2
df['feature1_log'] = np.log1p(df['feature1'])

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 65, 100])

# Encoding
df = pd.get_dummies(df, columns=['category'])

# Aggregation
df_grouped = df.groupby('category').agg({
    'price': 'mean',
    'quantity': 'sum'
})
```

## Data Augmentation Module Concepts

### Image Transformations
```python
from pipeline.augmentation import flip_image, rotate_image, adjust_brightness

# Flip horizontally
flipped = flip_image(image)

# Rotate by angle
rotated = rotate_image(image, angle=30)

# Adjust brightness
brightened = adjust_brightness(image, delta=0.3)
```

### Combining Augmentations
```python
def augment_batch(images, p=0.5):
    """Apply random augmentations to image batch"""
    augmented = []
    for img in images:
        if np.random.random() < p:
            img = flip_image(img)
        if np.random.random() < p:
            img = rotate_image(img, angle=np.random.randint(-15, 15))
        if np.random.random() < p:
            img = adjust_brightness(img, delta=np.random.uniform(-0.3, 0.3))
        augmented.append(img)
    return np.array(augmented)
```

## Best Practices

### Design Principles
1. **Modular**: Break pipeline into reusable components
2. **Reproducible**: Fix random seeds, document processes
3. **Scalable**: Use efficient algorithms and parallel processing
4. **Maintainable**: Clear variable names, good documentation
5. **Monitored**: Track data quality and pipeline performance

### Data Validation
```python
# Validate data shapes and types
assert X.shape[0] > 0, "Empty dataset"
assert X.shape[1] == expected_features, "Feature mismatch"

# Check for data quality
assert not X.isnull().any().any(), "Missing values present"
assert not np.isinf(X).any(), "Infinite values present"

# Verify distributions
print(X.describe())
print(X.corr())
```

### Common Pitfalls
- **Data Leakage**: Fitting scalers on full dataset before train/test split
- **Improper Ordering**: Transform before train/test split causes test leakage
- **Imbalanced Data**: Not accounting for class imbalance in classification
- **Feature Scaling**: Forgetting to scale test set with training scaler
- **Augmentation in Test**: Applying augmentation to test data reduces performance

### Optimization Tips
- Use Pandas `apply` with NumPy functions instead of Python loops
- Leverage `groupby` for efficient aggregation
- Use categorical data types to save memory
- Process data in chunks for very large datasets
- Parallelize augmentation using multiprocessing

## Common Pipeline Patterns

### Tabular Data (Pandas)
Raw CSV/JSON → Load → Clean → Encode → Scale → Feature Engineer → Train

### Image Data (Augmentation)
Raw Images → Load → Resize → Normalize → Augment → Train

### Time Series
Raw Data → Load → Handle Missing → Resample → Feature Engineer → Split by Time → Train

### Text Data
Raw Text → Load → Clean → Tokenize → Vectorize → Train

## Data Pipeline Checklist

- [ ] Data is loaded and exploratory analysis completed
- [ ] Missing values identified and handled
- [ ] Outliers detected and addressed
- [ ] Duplicates removed
- [ ] Data types are correct
- [ ] Categorical variables encoded
- [ ] Numerical features scaled
- [ ] Train/test split is stratified (for classification)
- [ ] Features are normalized using training data statistics
- [ ] Test set processed identically to training set
- [ ] Data leakage is prevented
- [ ] Pipeline is reproducible with fixed random seeds
- [ ] Documentation is complete

## Performance Monitoring

Track these metrics for pipeline health:
- **Data Quality**: Completeness, consistency, accuracy
- **Feature Distribution**: Mean, std, range, skewness
- **Train/Test Similarity**: Kullback-Leibler divergence
- **Pipeline Runtime**: Processing time per sample
- **Model Performance**: On training vs. validation data

## References

- Wickham, H. (2014). Tidy Data. Journal of Statistical Software
- McKinney, W. (2018). Python for Data Analysis (2nd Edition)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning (Chapter on Data)
