import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("customer_churn_data.csv")

# Encode categorical columns
label_encoder = LabelEncoder()
df["gender"] = label_encoder.fit_transform(df["gender"])  # Convert 'Male'/'Female' to 0/1

# Features and labels
X = df.drop(columns=["customer_id", "phone_no", "churn"])  # Exclude non-numeric columns
y = df["churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, "churn_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")  # Save the encoder to decode in prediction

print("Model trained and saved successfully!")
