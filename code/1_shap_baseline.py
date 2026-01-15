"""
Step 1: Train disease prediction model and generate SHAP explanations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import pickle
import os

# Get the project root directory (parent of 'code' folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

print("=" * 60)
print("STEP 1: SHAP BASELINE GENERATION")
print("=" * 60)

# 1. Load Heart Disease Dataset
print("\n[1/5] Loading dataset...")
df = pd.read_csv(os.path.join(DATA_DIR, 'heart_disease.csv'))

# Dataset columns (if using UCI format):
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                'ca', 'thal', 'target']

# If CSV doesn't have headers:
if df.shape[1] == 14 and 'age' not in df.columns:
    df.columns = column_names

# Clean data
df = df.replace('?', np.nan)
df = df.dropna()

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

print(f"   Dataset shape: {df.shape}")
print(f"   Features: {list(df.columns[:-1])}")

# 2. Prepare data
print("\n[2/5] Preparing train/test split...")
X = df.drop('target', axis=1)
y = df['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary: 0=no disease, 1=disease

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# 3. Train Random Forest Model
print("\n[3/5] Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"   Train Accuracy: {train_acc:.2%}")
print(f"   Test Accuracy: {test_acc:.2%}")

# 4. Generate SHAP values
print("\n[4/5] Generating SHAP explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle different SHAP output formats
# Newer SHAP versions return 3D arrays (samples, features, classes)
if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]  # For positive class (disease)
elif len(shap_values.shape) == 3:
    # 3D array: (samples, features, classes) - take class 1 (disease)
    shap_values_positive = shap_values[:, :, 1]
else:
    shap_values_positive = shap_values

print(f"   SHAP values shape: {shap_values_positive.shape}")

# 5. Extract top features globally
print("\n[5/5] Extracting feature importance...")
feature_importance = np.abs(shap_values_positive).mean(axis=0)
top_indices = np.argsort(feature_importance)[-5:][::-1]
# Convert to numpy array for indexing
feature_names = np.array(X.columns.tolist())
top_features = feature_names[top_indices].tolist()

print(f"   Top 5 Important Features:")
for i, (feat, imp) in enumerate(zip(top_features, feature_importance[top_indices]), 1):
    print(f"      {i}. {feat}: {imp:.4f}")

# 6. Save everything for next step
print("\n[6/6] Saving results...")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save model
with open(os.path.join(RESULTS_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Save SHAP explainer
with open(os.path.join(RESULTS_DIR, 'shap_explainer.pkl'), 'wb') as f:
    pickle.dump(explainer, f)

# Save SHAP values
with open(os.path.join(RESULTS_DIR, 'shap_values.pkl'), 'wb') as f:
    pickle.dump(shap_values_positive, f)

# Save test data
X_test.to_csv(os.path.join(RESULTS_DIR, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(RESULTS_DIR, 'y_test.csv'), index=False)

# Save summary JSON
import json
summary = {
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'features': X.columns.tolist(),
    'top_5_features': top_features,
    'feature_importance': {feat: float(feature_importance[idx]) 
                          for feat, idx in zip(top_features, top_indices)}
}

with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("✅ SHAP BASELINE COMPLETE!")
print("=" * 60)
print(f"\nSaved files:")
print("   - results/model.pkl")
print("   - results/shap_explainer.pkl")
print("   - results/shap_values.pkl")
print("   - results/X_test.csv")
print("   - results/y_test.csv")
print("   - results/summary.json")
print("\nNext: Run 'python code/2_generate_explanations.py'")
