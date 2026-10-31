from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

# 1. Locate the dataset
project_folder = Path(__file__).parent
dataset_path = project_folder / "dataset" / "Training Dataset.arff"

# 2. Load the ARFF dataset
data, metadata = arff.loadarff(dataset_path)
df = pd.DataFrame(data)

# 3. Convert byte/string values to numeric values
for column in df.columns:
    df[column] = pd.to_numeric(df[column])

# 4. Display duplicate information
print("Original dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

# 5. Separate features and target
X = df.drop(columns=["Result"])
y = df["Result"]

# 6. Convert labels:
# -1 = Phishing
#  1 = Legitimate
y = y.map({
    -1: 1,
     1: 0
})

# 7. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Display the results
print("\n========== DATA PREPARATION ==========")

print("Total records:", len(df))
print("Number of features:", X.shape[1])

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts().rename({
    1: "Phishing",
    0: "Legitimate"
}))

print("\nTesting target distribution:")
print(y_test.value_counts().rename({
    1: "Phishing",
    0: "Legitimate"
}))

print("\nMissing values in features:", X.isnull().sum().sum())
print("Missing values in target:", y.isnull().sum())

print("\nDataset preparation completed successfully!")
