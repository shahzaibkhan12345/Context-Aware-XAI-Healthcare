"""
Step 2: Generate context-aware explanations using Groq API
"""

import pandas as pd
import numpy as np
import json
import pickle
from groq import Groq
from dotenv import load_dotenv
import os

# Get directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Load environment variables
load_dotenv()

print("=" * 60)
print("STEP 2: CONTEXT-AWARE EXPLANATION GENERATION")
print("=" * 60)

# Initialize Groq client
print("\n[1/7] Initializing Groq API...")
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file!")

client = Groq(api_key=api_key)

# Choose model (Groq has several fast models)
MODEL = "llama-3.3-70b-versatile"  # Fast and capable
# Alternatives: "llama-3.1-8b-instant", "gemma2-9b-it"

print(f"   Using model: {MODEL}")

# Load test data
print("\n[2/7] Loading test data...")
X_test = pd.read_csv(os.path.join(RESULTS_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(RESULTS_DIR, 'y_test.csv'))

with open(os.path.join(RESULTS_DIR, 'shap_values.pkl'), 'rb') as f:
    shap_values = pickle.load(f)

with open(os.path.join(RESULTS_DIR, 'summary.json'), 'r') as f:
    summary = json.load(f)

top_features = summary['top_5_features']
print(f"   Loaded {len(X_test)} test samples")
print(f"   Top features: {', '.join(top_features[:3])}")

# Feature descriptions (for better explanations)
feature_descriptions = {
    'age': 'age in years',
    'sex': 'sex (1=male, 0=female)',
    'cp': 'chest pain type (0-3, higher=more severe)',
    'trestbps': 'resting blood pressure (mm Hg)',
    'chol': 'serum cholesterol (mg/dl)',
    'fbs': 'fasting blood sugar >120 mg/dl',
    'restecg': 'resting ECG results (0-2)',
    'thalach': 'maximum heart rate achieved',
    'exang': 'exercise induced angina (1=yes)',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'slope of peak exercise ST segment',
    'ca': 'number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'thalassemia blood disorder (3=normal, 6=fixed defect, 7=reversible defect)'
}

def format_patient_data(patient_dict):
    """Format patient data with descriptions"""
    formatted = []
    for key, value in patient_dict.items():
        desc = feature_descriptions.get(key, key)
        formatted.append(f"{key}={value:.1f} ({desc})")
    return ", ".join(formatted[:6])  # Limit to top 6 for brevity

# Define role-specific prompts
def create_prompt(patient_data, top_features_info, role):
    """Create role-specific prompts for LLM"""
    
    # Format patient data snippet
    patient_str = format_patient_data(patient_data)
    
    prompts = {
        'radiologist': f"""You are a radiologist explaining cardiovascular disease risk assessment to a medical team.

Patient Data: {patient_str}
Most Important Risk Factors (by SHAP analysis): {top_features_info}

Provide a HIGHLY TECHNICAL explanation using advanced medical terminology. Include:
- Specific cardiovascular imaging correlates and anatomical references
- Quantitative risk stratification using clinical scoring systems
- Detailed vessel disease burden assessment
- Technical SHAP value interpretation

IMPORTANT: Write at a GRADUATE-LEVEL medical complexity (Flesch-Kincaid Grade 14-16).
Use specialized radiology terminology, quantitative metrics, and precise clinical nomenclature.
Keep response under 110 words.""",

        'cardiologist': f"""You are a cardiologist explaining heart disease risk to another physician.

Patient Data: {patient_str}
Key Risk Factors (SHAP importance): {top_features_info}

Provide a COMPREHENSIVE CLINICAL explanation including:
- Pathophysiological mechanisms linking risk factors to outcomes
- Hemodynamic and electrophysiological significance
- Evidence-based clinical decision points
- Specific diagnostic and therapeutic recommendations

IMPORTANT: Write at MEDICAL SCHOOL level complexity (Flesch-Kincaid Grade 12-14).
Use clinical cardiology terminology but focus on actionable management.
Keep response under 100 words.""",

        'family_doctor': f"""You are a family doctor explaining heart disease risk to a colleague in primary care.

Patient Data: {patient_str}
Main Risk Factors: {top_features_info}

Provide a SIMPLE, PRACTICAL explanation that:
- AVOIDS complex medical jargon - use plain medical terms only
- Uses short sentences (under 15 words each)
- Focuses on practical next steps: lifestyle, referral, basic medications
- Prioritizes clarity over technical precision

CRITICAL: Write at HIGH SCHOOL reading level (Flesch-Kincaid Grade 8-10).
NO abbreviations, NO complex terminology, NO multi-syllable medical words.
Keep response under 60 words. Be brief and actionable.""",

        'patient': f"""You are explaining heart disease risk to a patient who has never studied medicine.

Key Risk Factors Found: {top_features_info}

Provide an EXTREMELY SIMPLE explanation that:
- Uses only everyday words a 10-year-old would understand
- Uses very short sentences (under 10 words each)
- Focuses on feelings and actions, not medical facts
- Is warm, encouraging, and supportive

CRITICAL: Write at ELEMENTARY SCHOOL reading level (Flesch-Kincaid Grade 5-6).
NO medical words at all. NO numbers or statistics.
Keep response under 45 words. Be kind and hopeful."""
    }
    
    return prompts[role]

# Generate explanation using Groq
def generate_explanation(patient_data, top_features_info, role):
    """Call Groq API to generate explanation"""
    
    prompt = create_prompt(patient_data, top_features_info, role)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical AI assistant specializing in explainable cardiovascular disease risk predictions. Adapt your language to the specified medical professional role."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=MODEL,
            temperature=0.7,
            max_tokens=180,
            top_p=0.9
        )
        
        return chat_completion.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"\n   ⚠️  Error generating explanation: {e}")
        return f"[Error: Unable to generate explanation - {str(e)[:50]}]"

# Main generation loop
print("\n[3/7] Generating explanations for test patients...")

roles = ['radiologist', 'cardiologist', 'family_doctor', 'patient']
all_results = []

# Generate for first 5 patients (for speed in 10-day sprint)
NUM_PATIENTS = 5

for idx in range(min(NUM_PATIENTS, len(X_test))):
    print(f"\n   Patient {idx+1}/{NUM_PATIENTS}:")
    
    # Get patient data
    patient_data = X_test.iloc[idx].to_dict()
    patient_shap = shap_values[idx]
    
    # Get top 3 features for THIS specific patient
    top_indices = np.argsort(np.abs(patient_shap))[-3:][::-1]
    top_patient_features = X_test.columns[top_indices].tolist()
    
    # Create human-readable summary
    top_features_info = ", ".join([
        f"{feat} ({feature_descriptions.get(feat, feat)}: {patient_data[feat]:.1f}, importance={abs(patient_shap[top_indices[i]]):.3f})"
        for i, feat in enumerate(top_patient_features)
    ])
    
    patient_result = {
        'patient_id': idx,
        'patient_data': patient_data,
        'actual_label': int(y_test.iloc[idx].values[0]),
        'predicted_label': 1 if sum(patient_shap) > 0 else 0,  # Simplified prediction
        'top_features': top_patient_features,
        'top_features_info': top_features_info,
        'explanations': {}
    }
    
    # Generate for each role
    for role in roles:
        print(f"      - {role}...", end=' ', flush=True)
        explanation = generate_explanation(patient_data, top_features_info, role)
        patient_result['explanations'][role] = explanation
        print("✓")
    
    all_results.append(patient_result)

# Save results
print("\n[4/7] Saving generated explanations...")
with open(os.path.join(RESULTS_DIR, 'explanations.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"   Saved: {os.path.join(RESULTS_DIR, 'explanations.json')}")

# Print example
print("\n[5/7] Example Output (Patient 1):")
print("-" * 60)
example = all_results[0]
print(f"Patient ID: {example['patient_id']}")
print(f"Actual Label: {'Heart Disease' if example['actual_label'] == 1 else 'No Heart Disease'}")
print(f"Top Risk Factors: {', '.join(example['top_features'])}")

print("\n" + "─" * 60)
for role in roles:
    print(f"\n🔹 {role.upper().replace('_', ' ')}:")
    print(f"   {example['explanations'][role]}")
    print(f"   (Word count: {len(example['explanations'][role].split())})")

print("\n" + "=" * 60)
print("✅ EXPLANATION GENERATION COMPLETE!")
print("=" * 60)
print(f"\nGenerated explanations:")
print(f"   - {len(all_results)} patients")
print(f"   - {len(roles)} roles per patient")
print(f"   - Total: {len(all_results) * len(roles)} unique explanations")
print(f"\nSaved to: {os.path.join(RESULTS_DIR, 'explanations.json')}")
print("\nNext: Run 'python code/3_evaluate.py'")
