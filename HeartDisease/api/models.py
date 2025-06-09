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
    model_path = 'api/models/heart_disease_model.joblib'
    scaler_path = 'api/models/scaler.joblib'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    else:
        raise FileNotFoundError("Model or scaler file not found")
if __name__ == '__main__':
    train_and_save_model()
