"""
Quick fix: Extract correct SHAP values from saved file
"""

import numpy as np
import pandas as pd
import pickle
import json
import os

# Get the project root directory (parent of 'code' folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

print("=" * 60)
print("FIXING SHAP VALUES")
print("=" * 60)

# Load saved SHAP values
print("\n[1/3] Loading saved SHAP values...")
with open(os.path.join(RESULTS_DIR, 'shap_values.pkl'), 'rb') as f:
    shap_values = pickle.load(f)

print(f"   Original shape: {shap_values.shape}")

# Fix: Take the positive class only (index 1)
if len(shap_values.shape) == 3:
    print("\n[2/3] Extracting positive class SHAP values...")
    shap_values_positive = shap_values[:, :, 1]  # Take class 1 (disease)
    print(f"   New shape: {shap_values_positive.shape}")
else:
    shap_values_positive = shap_values

# Load test data to get feature names
X_test = pd.read_csv(os.path.join(RESULTS_DIR, 'X_test.csv'))

# Extract top features
print("\n[3/3] Extracting feature importance...")
feature_importance = np.abs(shap_values_positive).mean(axis=0)
top_indices = np.argsort(feature_importance)[-5:][::-1]
top_features = X_test.columns[top_indices].tolist()

print(f"\n   Top 5 Important Features:")
for i, (feat, imp) in enumerate(zip(top_features, feature_importance[top_indices]), 1):
    print(f"      {i}. {feat}: {imp:.4f}")

# Save corrected SHAP values
print("\n[4/4] Saving corrected SHAP values...")
with open(os.path.join(RESULTS_DIR, 'shap_values.pkl'), 'wb') as f:
    pickle.dump(shap_values_positive, f)

# Update summary.json
with open(os.path.join(RESULTS_DIR, 'summary.json'), 'r') as f:
    summary = json.load(f)

summary['top_5_features'] = top_features
summary['feature_importance'] = {feat: float(feature_importance[idx]) 
                                  for feat, idx in zip(top_features, top_indices)}

with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("✅ SHAP VALUES FIXED!")
print("=" * 60)
print("\nCorrected files:")
print("   - results/shap_values.pkl (now 2D)")
print("   - results/summary.json (updated)")
print("\nNext: Run 'python code/2_generate_explanations.py'")
