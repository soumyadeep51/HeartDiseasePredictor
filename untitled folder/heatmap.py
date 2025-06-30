import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart.csv')

# Set style for seaborn
sns.set(style="whitegrid")

# 1. Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# 2. Feature Importance using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Convert categorical variables if present
X = pd.get_dummies(X, drop_first=True)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Plot Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Return the paths to generated files
"/mnt/data/correlation_heatmap.png", "/mnt/data/feature_importance.png"

