import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('heart.csv')

# Handle Cholesterol=0
df['Cholesterol'] = df['Cholesterol'].replace(0, df[df['Cholesterol'] != 0]['Cholesterol'].median())

# Define severity classes
def assign_severity(row):
    if row['HeartDisease'] == 0:
        return 0  # No Disease
    elif row['Oldpeak'] <= 1 and row['ExerciseAngina'] == 'N':
        return 1  # Mild
    elif (row['Oldpeak'] > 1 and row['Oldpeak'] <= 2) or row['ExerciseAngina'] == 'Y':
        return 2  # Moderate
    else:
        return 3  # Severe

df['Severity'] = df.apply(assign_severity, axis=1)

# Encode categorical variables
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and target
X = df_encoded.drop(['HeartDisease', 'Severity'], axis=1)
y = df_encoded['Severity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'feature_columns.pkl')

# Print accuracy on test set
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy:.4f}")