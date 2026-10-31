from pathlib import Path

import pandas as pd
import joblib

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Load dataset
project_folder = Path(__file__).parent
dataset_path = project_folder / "dataset" / "Training Dataset.arff"

data, metadata = arff.loadarff(dataset_path)
df = pd.DataFrame(data)

# 2. Convert values to numbers
for column in df.columns:
    df[column] = pd.to_numeric(df[column])

# 3. Separate input features and target
X = df.drop(columns=["Result"])

# 1 = Phishing, 0 = Legitimate
y = df["Result"].map({
    -1: 1,
     1: 0
})

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# 6. Train the model
print("Training the Random Forest model...")
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[1, 0],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print("Order: Phishing, Legitimate")
print(confusion_matrix(y_test, y_pred, labels=[1, 0]))

# 9. Save model and feature names
model_path = project_folder / "phishing_model.pkl"
features_path = project_folder / "feature_names.pkl"

joblib.dump(model, model_path)
joblib.dump(X.columns.tolist(), features_path)

print("\nModel saved as:", model_path.name)
print("Feature names saved as:", features_path.name)
print("\nTraining and evaluation completed!")
