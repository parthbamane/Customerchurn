import pandas as pd
import numpy as np
import random

# Generate sample dataset
num_samples = 1000
data = {
    "year": np.random.choice([2023, 2024], num_samples),
    "customer_id": range(1001, 1001 + num_samples),
    "phone_no": [random.randint(7000000000, 9999999999) for _ in range(num_samples)],
    "gender": np.random.choice(["Male", "Female"], num_samples),
    "age": np.random.randint(18, 60, num_samples),
    "no_of_days_subscribed": np.random.randint(30, 1000, num_samples),
    "multi_screen": np.random.choice([0, 1], num_samples),
    "mail_subscribed": np.random.choice([0, 1], num_samples),
    "weekly_mins_watched": np.random.randint(50, 5000, num_samples),
    "minimum_daily_mins": np.random.randint(5, 300, num_samples),
    "maximum_daily_mins": np.random.randint(50, 800, num_samples),
    "videos_watched": np.random.randint(10, 1000, num_samples),
    "maximum_days_inactive": np.random.randint(0, 30, num_samples),
    "customer_support_calls": np.random.randint(0, 10, num_samples),
    "churn": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),  # 20% churn
}

df = pd.DataFrame(data)
df["avg_daily_mins"] = df["weekly_mins_watched"] / 7
df.to_csv("customer_churn_data.csv", index=False)

print("Dataset created successfully!")
