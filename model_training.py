import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
import os

warnings.filterwarnings('ignore')


# Load dataset
def load_data():
    filepath = '/Users/parthbamane/Documents/project/data/customer_churn_data.csv'
    try:
        df = pd.read_csv(filepath)
        print("✅ Data loaded successfully. Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Missing values:\n", df.isna().sum())

        # Drop irrelevant columns
        df = df.drop(['year', 'customer_id', 'phone_no'], axis=1, errors='ignore')
        return df
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    if 'churn' not in df.columns:
        raise ValueError("'churn' column not found in dataset")

    print("\n🔍 Cleaning 'churn' column...")
    print("Original churn values:", df['churn'].unique())

    # Normalize churn values (handle both 0/1 and yes/no cases)
    if df['churn'].dtype == object:
        df['churn'] = df['churn'].astype(str).str.strip().str.lower().str.replace(r'[^a-z]', '', regex=True)
        df = df[df['churn'].isin(['yes', 'no'])]
        df['churn'] = df['churn'].map({'yes': 1, 'no': 0})
    else:
        df = df[df['churn'].isin([0, 1])]

    print("Cleaned churn values:\n", df['churn'].value_counts())

    if df.empty:
        raise ValueError("All rows removed after cleaning 'churn'. Check dataset.")

    X = df.drop('churn', axis=1)
    y = df['churn']

    categorical_features = ['gender', 'multi_screen', 'mail_subscribed']
    numerical_features = ['age', 'no_of_days_subscribed', 'weekly_mins_watched',
                          'minimum_daily_mins', 'maximum_daily_mins', 'videos_watched',
                          'maximum_days_inactive', 'customer_support_calls', 'avg_daily_mins']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor



def train_models(X, y, preprocessor):
    if len(X) == 0:
        raise ValueError("❌ No data available for training after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        try:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'model': pipeline
            }

            if results[name]['accuracy'] > best_score:
                best_score = results[name]['accuracy']
                best_model = name

        except Exception as e:
            print(f"\n❌ Error training {name}: {str(e)}")
            continue

    if not results:
        raise ValueError("❌ All models failed to train.")

    # Save metrics for each model
    for name, result in results.items():
        joblib.dump({
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1']
        }, f'models/{name.lower().replace(" ", "_")}_metrics.pkl')

    return results, best_model

# Main
def main():
    try:
        df = load_data()
        X, y, preprocessor = preprocess_data(df)

        print("\n🔎 Final data samples:")
        print(X.head())
        print(y.head())

        os.makedirs('models', exist_ok=True)  # <-- Ensure directory exists before saving anything

        results, best_model = train_models(X, y, preprocessor)

        for name, result in results.items():
            joblib.dump(result['model'], f'models/{name.lower().replace(" ", "_")}.pkl')

        joblib.dump(results[best_model]['model'], 'models/best_model.pkl')

        print("\n📊 Model Performance:")
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

        print(f"\n🏆 Best Model: {best_model} (Accuracy: {results[best_model]['accuracy']:.4f})")

    except Exception as e:
        print(f"\n💥 Fatal error in main: {str(e)}")
        raise



if __name__ == "__main__":
    main()
