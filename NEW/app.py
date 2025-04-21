from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
models = {
    'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
    'Random Forest': joblib.load('models/random_forest.pkl'),
    'Gradient Boosting': joblib.load('models/gradient_boosting.pkl'),
    'SVM': joblib.load('models/svm.pkl'),
    'Best Model': joblib.load('models/best_model.pkl')
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect form input
        form_data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'no_of_days_subscribed': float(request.form['no_of_days_subscribed']),
            'multi_screen': request.form['multi_screen'],
            'mail_subscribed': request.form['mail_subscribed'],
            'weekly_mins_watched': float(request.form['weekly_mins_watched']),
            'minimum_daily_mins': float(request.form['minimum_daily_mins']),
            'maximum_daily_mins': float(request.form['maximum_daily_mins']),
            'videos_watched': float(request.form['videos_watched']),
            'maximum_days_inactive': float(request.form['maximum_days_inactive']),
            'customer_support_calls': float(request.form['customer_support_calls']),
            'avg_daily_mins': float(request.form['avg_daily_mins'])
        }

        input_df = pd.DataFrame([form_data])

        predictions = {}
        accuracies = {}
        f1_scores = {}
        error_percentages = {}

        # Loop through models
        for name, model in models.items():
            if name != 'Best Model':
                try:
                    proba = model.predict_proba(input_df)[0][1]
                    pred = model.predict(input_df)[0]
                    predictions[name] = {
                        'prediction': 'yes' if pred == 1 else 'no',
                        'probability': f"{proba*100:.2f}%",
                        'probability_value': proba
                    }

                    # Load model metrics
                    metric_path = f'models/{name.lower().replace(" ", "_")}_metrics.pkl'
                    metrics = joblib.load(metric_path)

                    accuracies[name] = f"{metrics.get('accuracy', 0) * 100:.2f}%"
                    f1_scores[name] = f"{metrics.get('f1_score', 0) * 100:.2f}%"
                    error_percentages[name] = f"{metrics.get('error_percentage', 0):.2f}%"

                except Exception as e:
                    predictions[name] = {
                        'prediction': 'error',
                        'probability': 'error',
                        'probability_value': 0
                    }
                    accuracies[name] = 'error'
                    f1_scores[name] = 'error'
                    error_percentages[name] = 'error'

        # Determine the best model by highest probability
        best_model_name = None
        best_model_prob = -1

        for name, pred in predictions.items():
            try:
                prob = float(pred['probability'].strip('%')) / 100
                if prob > best_model_prob:
                    best_model_prob = prob
                    best_model_name = name
            except:
                continue

        if best_model_name:
            best_pred = predictions[best_model_name]
            predictions['Best Model'] = {
                'prediction': best_pred['prediction'],
                'probability': best_pred['probability'],
                'model_name': best_model_name
            }
        else:
            predictions['Best Model'] = {
                'prediction': 'no consensus',
                'probability': 'N/A',
                'model_name': 'None'
            }

        return render_template('index.html',
                               predictions=predictions,
                               accuracies=accuracies,
                               f1_scores=f1_scores,
                               error_percentages=error_percentages,
                               form_data=form_data,
                               show_results=True)

    return render_template('index.html', show_results=False)

if __name__ == '__main__':
    app.run(debug=True)
