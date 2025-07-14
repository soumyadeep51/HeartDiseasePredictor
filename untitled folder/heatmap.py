import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart.csv')

# Set style for seaborn
sns.set(style="whitegrid")

# 1. Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation = df.corr(numeric_only=True)  # Only numeric columns
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.savefig("correlation_heatmap.jpg", format='jpg', dpi=300)
plt.close()

# 2. Feature Importance using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Prepare data
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Feature importances
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.savefig("feature_importance.jpg", format='jpg', dpi=300)
plt.close()

# Output paths
print("Plots saved as JPG:")
print("/mnt/data/correlation_heatmap.jpg")
print("/mnt/data/feature_importance.jpg")

