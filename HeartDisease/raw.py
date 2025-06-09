# models/heart_disease_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_prepare_data():
    # Load the provided heart.csv dataset
    data = pd.read_csv('heart.csv')
    
    # Encode categorical variables
    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])  # M: 1, F: 0
    data['ChestPainType'] = le.fit_transform(data['ChestPainType'])  # ATA: 0, NAP: 1, ASY: 2, TA: 3
    data['RestingECG'] = le.fit_transform(data['RestingECG'])  # Normal: 1, ST: 2, LVH: 0
    data['ExerciseAngina'] = le.fit_transform(data['ExerciseAngina'])  # N: 0, Y: 1
    data['ST_Slope'] = le.fit_transform(data['ST_Slope'])  # Up: 2, Flat: 1, Down: 0
    
    # Define features and target
    features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    X = data[features]
    y = data['HeartDisease']
    
    return X, y, features

def calculate_severity(oldpeak):
    # Severity score based on Oldpeak (ST depression)
    # Oldpeak ranges in dataset: -2.6 to 6.2
    # Normalize to 0-100 scale: <0 (low), 0-2 (moderate), >2 (high)
    if oldpeak < 0:
        return 10  # Low severity for negative Oldpeak
    elif 0 <= oldpeak <= 2:
        return int(10 + (oldpeak / 2) * 40)  # Linear scaling from 10 to 50
    else:
        return int(50 + ((oldpeak - 2) / 4.2) * 50)  # Linear scaling from 50 to 100

def train_and_save_model():
    X, y, features = load_and_prepare_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save model and scaler
    model_path = 'models/heart_disease_model.joblib'
    scaler_path = 'models/scaler.joblib'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return rf_model, scaler, features

def load_model_and_scaler():
    model_path = 'models/heart_disease_model.joblib'
    scaler_path = 'models/scaler.joblib'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    else:
        raise FileNotFoundError("Model or scaler file not found")

# api/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
from .heart_disease_model import load_model_and_scaler, calculate_severity

@csrf_exempt
def predict_heart_disease(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            features = np.array([
                float(data['Age']),
                float(data['Sex']),  # 1: Male, 0: Female
                float(data['ChestPainType']),  # 0: ATA, 1: NAP, 2: ASY, 3: TA
                float(data['RestingBP']),
                float(data['Cholesterol']),
                float(data['FastingBS']),
                float(data['RestingECG']),  # 1: Normal, 2: ST, 0: LVH
                float(data['MaxHR']),
                float(data['ExerciseAngina']),  # 0: No, 1: Yes
                float(data['Oldpeak']),
                float(data['ST_Slope'])  # 2: Up, 1: Flat, 0: Down
            ]).reshape(1, -1)
            
            model, scaler = load_model_and_scaler()
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled).tolist()
            severity = calculate_severity(float(data['Oldpeak']))
            
            return JsonResponse({
                'prediction': int(prediction[0]),  # 0: No disease, 1: Disease
                'probability': probability[0],  # [P(No disease), P(Disease)]
                'severity': severity  # Severity score (0-100)
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

# api/urls.py
from django.urls import path
from .views import predict_heart_disease

urlpatterns = [
    path('predict/', predict_heart_disease, name='predict_heart_disease'),
]

# django_project/settings.py (partial - add to your existing settings)
INSTALLED_APPS = [
    # ... other apps ...
    'api',
    'rest_framework',
    'corsheaders',
]

MIDDLEWARE = [
    # ... other middleware ...
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # React frontend URL
]

# Train model when script is run directly
if __name__ == '__main__':
    train_and_save_model()