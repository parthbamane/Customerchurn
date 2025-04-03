import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE  # Handle class imbalance
import joblib

# Load dataset
df = pd.read_csv("customer_churn_data.csv")

# Encode categorical columns
label_encoder = LabelEncoder()
df["gender"] = label_encoder.fit_transform(df["gender"])  # Convert 'Male'/'Female' to 0/1

# Features and labels
X = df.drop(columns=["customer_id", "phone_no", "churn"], errors="ignore")  # Exclude unnecessary columns
y = df["churn"]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split (Stratify for balanced class distribution)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Apply Standard Scaling for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga'),  # Better solver for large datasets
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, eval_metric='logloss', random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42)
}

# Train and save models
for name, model in models.items():
    try:
        print(f"Training {name}...")
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)  # Use scaled data for LR
        else:
            model.fit(X_train, y_train)

        joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")
    except Exception as e:
        print(f"❌ Error training {name}: {e}")

# Save the label encoder and scaler
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ All models trained and saved successfully!")
