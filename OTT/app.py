from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model and encoder
model_path = "random_forest_model.pkl"
encoder_path = "label_encoder.pkl"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("Trained model or label encoder not found! Run train_model.py first.")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract user input
        gender = request.form["gender"]
        gender_encoded = label_encoder.transform([gender])[0]  # Encode 'Male' or 'Female'

        # Convert input values to the expected format
        data = np.array([
            int(request.form["year"]),
            gender_encoded,
            int(request.form["age"]),
            int(request.form["no_of_days_subscribed"]),
            int(request.form["multi_screen"]),
            int(request.form["mail_subscribed"]),
            int(request.form["weekly_mins_watched"]),
            int(request.form["minimum_daily_mins"]),
            int(request.form["maximum_daily_mins"]),
            int(request.form["videos_watched"]),
            int(request.form["maximum_days_inactive"]),
            int(request.form["customer_support_calls"]),
            float(request.form["avg_daily_mins"])
        ]).reshape(1, -1)  # Reshape for model prediction

        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1] * 100  # Get churn probability

        # Prepare response
        result = {
            "churn": "Yes" if prediction == 1 else "No",
            "probability": f"{probability:.2f}%"
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
