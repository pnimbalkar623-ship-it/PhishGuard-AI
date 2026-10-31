from pathlib import Path
import pandas as pd
from scipy.io import arff

# Locate dataset
dataset_path = (
    Path(__file__).parent
    / "dataset"
    / "Training Dataset.arff"
)

# Check file
if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {dataset_path}"
    )

# Load ARFF dataset
data, metadata = arff.loadarff(dataset_path)

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Decode byte labels if present
for column in df.select_dtypes(
    include=["object"]
).columns:
    df[column] = df[column].apply(
        lambda value: (
            value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )
    )

print("\n========== DATASET OVERVIEW ==========")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df["Result"].value_counts())

print("\nTarget distribution in percentage:")
print(
    df["Result"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFeature value ranges:")
print(df.nunique())

print("\n========== ANALYSIS COMPLETE ==========")