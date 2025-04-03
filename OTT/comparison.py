# from google.colab import files
# uploaded=files.upload()
# print(uploaded)

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("customer_churn_data.csv")

# Load trained model & label encoder
model = joblib.load("random_forest_model.pkl")
# model = joblib.load("decision_tree_model.pkl")
# model = joblib.load("xgboost_model.pkl")
# model = joblib.load("logistic_regression_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Ensure the gender column is encoded correctly
if "gender" in df.columns:
    df["gender"] = label_encoder.transform(df["gender"])  # Convert categorical to numerical

# Define features and target
X = df.drop(columns=["customer_id", "phone_no", "churn"], errors="ignore")  # Exclude non-numeric columns safely
y = df["churn"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display classification report with zero_division=1 to avoid warnings
print(classification_report(y_test, y_pred, zero_division=1))
