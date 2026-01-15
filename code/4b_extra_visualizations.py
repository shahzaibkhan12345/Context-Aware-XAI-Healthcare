"""
Step 4b: Additional publication-ready visualizations for IEEE paper
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

os.makedirs(FIGURES_DIR, exist_ok=True)

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

print("=" * 60)
print("ADDITIONAL VISUALIZATIONS FOR IEEE PAPER")
print("=" * 60)

# Load data
print("\n[Loading Data]...")
summary_df = pd.read_csv(os.path.join(RESULTS_DIR, 'summary_metrics.csv'), index_col=0)
detailed_df = pd.read_csv(os.path.join(RESULTS_DIR, 'detailed_metrics.csv'))

with open(os.path.join(RESULTS_DIR, 'explanations.json'), 'r') as f:
    explanations = json.load(f)

with open(os.path.join(RESULTS_DIR, 'summary.json'), 'r') as f:
    model_summary = json.load(f)

roles = ['radiologist', 'cardiologist', 'family_doctor', 'patient']
role_labels = ['Radiologist', 'Cardiologist', 'Family Doctor', 'Patient']
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

# ==============================================================================
# FIGURE 6: Radar Chart - Multi-dimensional Role Comparison
# ==============================================================================
print("\n[1/8] Creating Radar Chart...")

categories = ['Word Count', 'Readability\n(inverted)', 'Medical\nTerms', 'Actionability', 'Sentences']
N = len(categories)

# Normalize metrics to 0-1 scale for radar
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 0.001)

word_norm = normalize(summary_df['avg_word_count'].values)
fk_norm = 1 - normalize(summary_df['avg_flesch_kincaid'].values)  # Invert: lower grade = more readable
med_norm = normalize(summary_df['avg_medical_terms'].values)
act_norm = normalize(summary_df['avg_actionability'].values)
sent_norm = normalize(summary_df['avg_sentence_count'].values)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, (role, label) in enumerate(zip(roles, role_labels)):
    values = [word_norm[i], fk_norm[i], med_norm[i], act_norm[i], sent_norm[i]]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
    ax.fill(angles, values, alpha=0.15, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.title('Multi-Dimensional Role Comparison\n(Normalized Metrics)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure6_radar_chart.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: figure6_radar_chart.png")

# ==============================================================================
# FIGURE 7: Heatmap - Metrics Correlation by Role
# ==============================================================================
print("\n[2/8] Creating Heatmap...")

fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = summary_df[['avg_word_count', 'avg_flesch_kincaid', 'avg_medical_terms', 
                            'avg_reading_ease', 'avg_actionability']].T
heatmap_data.columns = role_labels
heatmap_data.index = ['Word Count', 'Grade Level', 'Medical Terms', 'Reading Ease', 'Actionability']

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Metric Value'})
ax.set_title('Explanation Metrics Heatmap by Role', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure7_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: figure7_heatmap.png")

# ==============================================================================
# FIGURE 8: SHAP Feature Importance (Global)
# ==============================================================================
print("\n[3/8] Creating SHAP Feature Importance Bar Chart...")

fig, ax = plt.subplots(figsize=(10, 6))

features = model_summary['top_5_features']
importance = [model_summary['feature_importance'][f] for f in features]

bars = ax.barh(features, importance, color='#3498DB', edgecolor='black')
ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Top 5 Features by SHAP Importance\n(Heart Disease Prediction)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for bar, val in zip(bars, importance):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure8_shap_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: figure8_shap_importance.png")

# ==============================================================================
# FIGURE 9: Box Plot - Distribution Across Patients
# ==============================================================================
print("\n[4/8] Creating Box Plots...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Word Count Distribution
sns.boxplot(data=detailed_df, x='role', y='word_count', ax=axes[0], palette=colors, order=roles)
axes[0].set_xticklabels(role_labels, rotation=15, ha='right')
axes[0].set_title('Word Count Distribution', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Words')

# Readability Distribution
sns.boxplot(data=detailed_df, x='role', y='flesch_kincaid_grade', ax=axes[1], palette=colors, order=roles)
axes[1].set_xticklabels(role_labels, rotation=15, ha='right')
axes[1].set_title('Readability (Grade Level)', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Grade Level')

# Medical Terms Distribution
sns.boxplot(data=detailed_df, x='role', y='medical_term_count', ax=axes[2], palette=colors, order=roles)
axes[2].set_xticklabels(role_labels, rotation=15, ha='right')
axes[2].set_title('Medical Terminology', fontweight='bold')
axes[2].set_xlabel('')
axes[2].set_ylabel('Term Count')

plt.suptitle('Metric Distribution Across Patients by Role', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure9_boxplots.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: figure9_boxplots.png")

# ==============================================================================
# FIGURE 10: Complexity Gradient (Line + Point)
# ==============================================================================
print("\n[5/8] Creating Complexity Gradient Chart...")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(roles))
y_grade = summary_df['avg_flesch_kincaid'].values
y_terms = summary_df['avg_medical_terms'].values

ax.plot(x, y_grade, 'o-', color='#E74C3C', linewidth=2.5, markersize=12, label='Grade Level')
ax2 = ax.twinx()
ax2.plot(x, y_terms, 's--', color='#3498DB', linewidth=2.5, markersize=12, label='Medical Terms')

ax.set_xticks(x)
ax.set_xticklabels(role_labels)
ax.set_ylabel('Flesch-Kincaid Grade Level', color='#E74C3C', fontsize=12, fontweight='bold')
ax2.set_ylabel('Medical Term Count', color='#3498DB', fontsize=12, fontweight='bold')
ax.set_xlabel('Clinician Role (Specialist → General)', fontsize=12, fontweight='bold')
ax.set_title('Complexity Gradient: Technical → Simple', fontsize=14, fontweight='bold', pad=15)

# Add arrow annotation
ax.annotate('', xy=(3, y_grade[3]), xytext=(0, y_grade[0]),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
ax.text(1.5, (y_grade[0] + y_grade[3])/2 + 1, 'Complexity\nReduction', ha='center', fontsize=10, color='gray')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure10_complexity_gradient.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: figure10_complexity_gradient.png")

# ==============================================================================
# FIGURE 11: Reading Ease vs Word Count (Scatter)
# ==============================================================================
print("\n[6/8] Creating Scatter Plot...")

fig, ax = plt.subplots(figsize=(8, 6))

for i, (role, label) in enumerate(zip(roles, role_labels)):
    role_data = detailed_df[detailed_df['role'] == role]
    ax.scatter(role_data['word_count'], role_data['flesch_reading_ease'], 
               s=100, c=colors[i], label=label, alpha=0.8, edgecolors='black')

ax.set_xlabel('Word Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Flesch Reading Ease Score', fontsize=12, fontweight='bold')
ax.set_title('Reading Ease vs. Explanation Length', fontsize=14, fontweight='bold', pad=15)
ax.legend()
ax.grid(alpha=0.3)

# Add regions
ax.axhline(y=60, color='green', linestyle='--', alpha=0.5)
ax.axhline(y=30, color='red', linestyle='--', alpha=0.5)
ax.text(70, 65, 'Easy to Read', color='green', fontsize=9)
ax.text(70, 25, 'Difficult', color='red', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure11_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: figure11_scatter.png")

# ==============================================================================
# FIGURE 12: Framework Architecture Diagram (Conceptual)
# ==============================================================================
print("\n[7/8] Creating Framework Diagram...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Draw boxes for framework components
boxes = [
    {'pos': (0.1, 0.7), 'text': 'Patient\nData', 'color': '#3498DB'},
    {'pos': (0.3, 0.7), 'text': 'ML Model\n(Random Forest)', 'color': '#E74C3C'},
    {'pos': (0.5, 0.7), 'text': 'SHAP\nExplainer', 'color': '#9B59B6'},
    {'pos': (0.7, 0.7), 'text': 'Context-Aware\nLLM Adapter', 'color': '#2ECC71'},
    {'pos': (0.7, 0.4), 'text': 'Role\nDetection', 'color': '#F39C12'},
    {'pos': (0.9, 0.8), 'text': 'Radiologist\nExplanation', 'color': '#E74C3C'},
    {'pos': (0.9, 0.6), 'text': 'Cardiologist\nExplanation', 'color': '#3498DB'},
    {'pos': (0.9, 0.4), 'text': 'Family Doctor\nExplanation', 'color': '#2ECC71'},
    {'pos': (0.9, 0.2), 'text': 'Patient\nExplanation', 'color': '#F39C12'},
]

for box in boxes:
    rect = plt.Rectangle((box['pos'][0]-0.07, box['pos'][1]-0.08), 0.14, 0.16, 
                          facecolor=box['color'], alpha=0.3, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(box['pos'][0], box['pos'][1], box['text'], ha='center', va='center', 
            fontsize=10, fontweight='bold')

# Draw arrows
arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
ax.annotate('', xy=(0.23, 0.7), xytext=(0.17, 0.7), arrowprops=arrow_style)
ax.annotate('', xy=(0.43, 0.7), xytext=(0.37, 0.7), arrowprops=arrow_style)
ax.annotate('', xy=(0.63, 0.7), xytext=(0.57, 0.7), arrowprops=arrow_style)
ax.annotate('', xy=(0.7, 0.56), xytext=(0.7, 0.62), arrowprops=arrow_style)
ax.annotate('', xy=(0.83, 0.8), xytext=(0.77, 0.75), arrowprops=arrow_style)
ax.annotate('', xy=(0.83, 0.6), xytext=(0.77, 0.65), arrowprops=arrow_style)
ax.annotate('', xy=(0.83, 0.4), xytext=(0.77, 0.45), arrowprops=arrow_style)
ax.annotate('', xy=(0.83, 0.25), xytext=(0.77, 0.35), arrowprops=arrow_style)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Context-Aware XAI Framework Architecture', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure12_framework.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved: figure12_framework.png")

# ==============================================================================
# FIGURE 13: Metric Reduction Summary (Percentage Change)
# ==============================================================================
print("\n[8/8] Creating Reduction Summary Chart...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate percentage reduction from Radiologist to Patient
rad = summary_df.loc['radiologist']
pat = summary_df.loc['patient']

reductions = {
    'Word Count': ((rad['avg_word_count'] - pat['avg_word_count']) / rad['avg_word_count']) * 100,
    'Grade Level': ((rad['avg_flesch_kincaid'] - pat['avg_flesch_kincaid']) / rad['avg_flesch_kincaid']) * 100,
    'Medical Terms': ((rad['avg_medical_terms'] - pat['avg_medical_terms']) / rad['avg_medical_terms']) * 100,
}

# Reading ease is inverse (increase is good)
reductions['Reading Ease\n(Increase)'] = ((pat['avg_reading_ease'] - rad['avg_reading_ease']) / rad['avg_reading_ease']) * 100

labels = list(reductions.keys())
values = list(reductions.values())
bar_colors = ['#E74C3C' if v > 0 else '#2ECC71' for v in values]
bar_colors[3] = '#2ECC71'  # Reading ease increase is positive

bars = ax.barh(labels, values, color=bar_colors, edgecolor='black')

for bar, val in zip(bars, values):
    ax.text(val + (2 if val > 0 else -2), bar.get_y() + bar.get_height()/2, 
            f'{val:.0f}%', va='center', fontsize=11, fontweight='bold',
            ha='left' if val > 0 else 'right')

ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel('Percentage Change (%)', fontsize=12, fontweight='bold')
ax.set_title('Adaptation Impact: Radiologist → Patient\n(Context-Aware Transformation)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(-50, 120)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure13_reduction_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: figure13_reduction_summary.png")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 60)
print("✅ ADDITIONAL VISUALIZATIONS COMPLETE!")
print("=" * 60)
print("\nCreated 8 additional publication-ready figures:")
print("   6. figure6_radar_chart.png - Multi-dimensional role comparison")
print("   7. figure7_heatmap.png - Metrics correlation heatmap")
print("   8. figure8_shap_importance.png - SHAP feature importance")
print("   9. figure9_boxplots.png - Distribution across patients")
print("  10. figure10_complexity_gradient.png - Technical→Simple gradient")
print("  11. figure11_scatter.png - Reading ease vs word count")
print("  12. figure12_framework.png - Framework architecture diagram")
print("  13. figure13_reduction_summary.png - Adaptation impact summary")
print(f"\nAll figures saved to: {FIGURES_DIR}")
print("\n📊 These are ready for your IEEE paper sections!")
