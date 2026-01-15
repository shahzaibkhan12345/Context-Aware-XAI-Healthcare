"""
Step 4: Create publication-ready visualizations for IEEE paper
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no Tk required)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

print("=" * 60)
print("STEP 4: VISUALIZATION GENERATION")
print("=" * 60)

# Load data
print("\n[1/6] Loading evaluation metrics...")
summary_df = pd.read_csv(os.path.join(RESULTS_DIR, 'summary_metrics.csv'), index_col=0)
detailed_df = pd.read_csv(os.path.join(RESULTS_DIR, 'detailed_metrics.csv'))

with open(os.path.join(RESULTS_DIR, 'explanations.json'), 'r') as f:
    explanations = json.load(f)

print("   Data loaded successfully ✓")

roles = ['radiologist', 'cardiologist', 'family_doctor', 'patient']
role_labels = ['Radiologist', 'Cardiologist', 'Family Doctor', 'Patient']

# FIGURE 1: Word Count by Role
print("\n[2/6] Creating Figure 1: Word Count by Role...")

fig, ax = plt.subplots(figsize=(8, 5))
word_counts = summary_df['avg_word_count'].values
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

bars = ax.bar(role_labels, word_counts, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Average Word Count', fontsize=12, fontweight='bold')
ax.set_xlabel('Clinician Role', fontsize=12, fontweight='bold')
ax.set_title('Explanation Length by Role', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(word_counts) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure1_word_count.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {os.path.join(FIGURES_DIR, 'figure1_word_count.png')}")

# FIGURE 2: Readability (Flesch-Kincaid Grade Level)
print("\n[3/6] Creating Figure 2: Readability Scores...")

fig, ax = plt.subplots(figsize=(8, 5))
fk_grades = summary_df['avg_flesch_kincaid'].values

bars = ax.bar(role_labels, fk_grades, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add reference lines
ax.axhline(y=12, color='red', linestyle='--', alpha=0.5, label='High School (Grade 12)')
ax.axhline(y=8, color='orange', linestyle='--', alpha=0.5, label='Middle School (Grade 8)')

ax.set_ylabel('Flesch-Kincaid Grade Level', fontsize=12, fontweight='bold')
ax.set_xlabel('Clinician Role', fontsize=12, fontweight='bold')
ax.set_title('Explanation Readability by Role', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(fk_grades) * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure2_readability.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {os.path.join(FIGURES_DIR, 'figure2_readability.png')}")

# FIGURE 3: Medical Terminology Usage
print("\n[4/6] Creating Figure 3: Medical Terminology Usage...")

fig, ax = plt.subplots(figsize=(8, 5))
med_terms = summary_df['avg_medical_terms'].values

bars = ax.bar(role_labels, med_terms, color=colors, edgecolor='black', linewidth=1.2)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Average Medical Terms per Explanation', fontsize=12, fontweight='bold')
ax.set_xlabel('Clinician Role', fontsize=12, fontweight='bold')
ax.set_title('Medical Terminology Density by Role', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(med_terms) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure3_medical_terms.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {os.path.join(FIGURES_DIR, 'figure3_medical_terms.png')}")

# FIGURE 4: Comparison Table (as image for paper)
print("\n[5/6] Creating Figure 4: Metrics Summary Table...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = [
    ['Role', 'Avg Words', 'Grade Level', 'Medical Terms', 'Actionability'],
]

for role, label in zip(roles, role_labels):
    row = [
        label,
        f"{summary_df.loc[role, 'avg_word_count']:.0f}",
        f"{summary_df.loc[role, 'avg_flesch_kincaid']:.1f}",
        f"{summary_df.loc[role, 'avg_medical_terms']:.1f}",
        f"{summary_df.loc[role, 'avg_actionability']:.1f}"
    ]
    table_data.append(row)

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.15, 0.15, 0.20, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#4ECDC4')
    cell.set_text_props(weight='bold', color='white')

# Color rows alternately
for i in range(1, 5):
    for j in range(5):
        cell = table[(i, j)]
        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

plt.title('Summary of Explanation Metrics by Role', 
          fontsize=14, fontweight='bold', pad=20)

plt.savefig(os.path.join(FIGURES_DIR, 'figure4_metrics_table.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved: {os.path.join(FIGURES_DIR, 'figure4_metrics_table.png')}")

# FIGURE 5: Combined Multi-metric Comparison
print("\n[6/6] Creating Figure 5: Multi-Metric Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Word Count
axes[0, 0].bar(role_labels, word_counts, color=colors, edgecolor='black')
axes[0, 0].set_title('A) Word Count', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Words')
axes[0, 0].grid(axis='y', alpha=0.3)

# Subplot 2: Readability
axes[0, 1].bar(role_labels, fk_grades, color=colors, edgecolor='black')
axes[0, 1].set_title('B) Readability Grade', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Grade Level')
axes[0, 1].grid(axis='y', alpha=0.3)

# Subplot 3: Medical Terms
axes[1, 0].bar(role_labels, med_terms, color=colors, edgecolor='black')
axes[1, 0].set_title('C) Medical Terminology', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Term Count')
axes[1, 0].grid(axis='y', alpha=0.3)

# Subplot 4: Actionability
actionability = summary_df['avg_actionability'].values
axes[1, 1].bar(role_labels, actionability, color=colors, edgecolor='black')
axes[1, 1].set_title('D) Actionability Score', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Context-Aware Explanation Metrics Across Roles', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

plt.savefig(os.path.join(FIGURES_DIR, 'figure5_combined.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {os.path.join(FIGURES_DIR, 'figure5_combined.png')}")

# Generate LaTeX table for paper
print("\n[BONUS] Generating LaTeX table for IEEE paper...")
latex_table = summary_df[['avg_word_count', 'avg_flesch_kincaid', 
                          'avg_medical_terms', 'avg_actionability']].round(2)
latex_table.index = role_labels
latex_output = latex_table.to_latex()

with open(os.path.join(RESULTS_DIR, 'latex_table.txt'), 'w') as f:
    f.write(latex_output)
print(f"   Saved: {os.path.join(RESULTS_DIR, 'latex_table.txt')}")

print("\n" + "=" * 60)
print("✅ VISUALIZATION COMPLETE!")
print("=" * 60)
print(f"\nCreated {5} publication-ready figures:")
print(f"   1. figure1_word_count.png")
print(f"   2. figure2_readability.png")
print(f"   3. figure3_medical_terms.png")
print(f"   4. figure4_metrics_table.png")
print(f"   5. figure5_combined.png (multi-panel)")
print(f"\nAll figures saved to: {FIGURES_DIR}")
print("\n📊 These figures are ready for your IEEE paper!")
print("\nNext: Start writing your paper (Days 3-5)")
