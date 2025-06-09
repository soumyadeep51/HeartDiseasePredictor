import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')

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

# Initialize models
models = {
    'SVM': SVC(kernel='linear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
}

# Store results
accuracies = {}
confusion_matrices = {}

# Train and evaluate models
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracies[name] = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrices[name], annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Mild', 'Moderate', 'Severe'],
                yticklabels=['No Disease', 'Mild', 'Moderate', 'Severe'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Plot feature importance for Decision Tree and Random Forest
feature_names = X.columns
for name in ['Decision Tree', 'Random Forest']:
    model = models[name]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center', color='skyblue')
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.title(f'Feature Importance - {name}')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'plots/feature_importance_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Plot accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.savefig('plots/accuracy_comparison.png')
plt.close()

# Print accuracies
print("Model Accuracies:")
for name, acc in accuracies.items():
    print(f"{name}: {acc:.4f}")