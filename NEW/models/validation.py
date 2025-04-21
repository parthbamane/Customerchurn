# Example (in Python script or notebook)
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"/Users/parthbamane/Documents/project/data/customer_churn_data.csv")  # Your full labeled data
# Assuming 'churn' is the correct name of the target column
X = df.drop("churn", axis=1)
y = df["churn"]

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a copy of the validation data with the target column added
val_data = X_val.copy()
val_data["churn"] = y_val  # Using 'churn' as the target column name

# Save the validation data
val_data.to_csv("models/validation_data.csv", index=False)
