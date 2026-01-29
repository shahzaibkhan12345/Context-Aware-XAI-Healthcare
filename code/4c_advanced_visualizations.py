"""
Step 4c: Visualize Advanced Evaluation Results (Grounding & Baseline)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Get directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

print("=" * 60)
print("STEP 4c: ADVANCED EVALUATION VISUALIZATIONS")
print("=" * 60)

# Load results
with open(os.path.join(RESULTS_DIR, 'advanced_evaluation.json'), 'r') as f:
    advanced_results = json.load(f)

grounding_df = pd.read_csv(os.path.join(RESULTS_DIR, 'grounding_analysis.csv'))
comparison_df = pd.read_csv(os.path.join(RESULTS_DIR, 'baseline_comparison.csv'))

roles = ['radiologist', 'cardiologist', 'family_doctor', 'patient']
role_labels = ['Radiologist', 'Cardiologist', 'Family Doctor', 'Patient']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# =============================================================================
# FIGURE 1: Feature Grounding Scores
# =============================================================================
print("\n[1/3] Creating Feature Grounding figure...")

fig, ax = plt.subplots(figsize=(10, 6))

grounding_scores = [advanced_results['grounding_by_role'][role]['mean_grounding'] * 100 
                   for role in roles]
grounding_stds = [advanced_results['grounding_by_role'][role]['std_grounding'] * 100 
                 for role in roles]

bars = ax.bar(role_labels, grounding_scores, color=colors, edgecolor='black', linewidth=1.5)

# Add error bars
ax.errorbar(role_labels, grounding_scores, yerr=grounding_stds, 
           fmt='none', color='black', capsize=5, capthick=2)

# Add value labels
for bar, score in zip(bars, grounding_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
           f'{score:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add threshold lines
ax.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='High Faithfulness (80%)')
ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Moderate Faithfulness (50%)')

ax.set_ylabel('Feature Grounding Score (%)', fontsize=12)
ax.set_xlabel('Clinician Role', fontsize=12)
ax.set_title('LLM Explanation Faithfulness to SHAP Features\n(Higher = More SHAP Features Mentioned)', 
            fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure14_grounding.png'), dpi=150, bbox_inches='tight')
print(f"   Saved: figure14_grounding.png")
plt.close()

# =============================================================================
# FIGURE 2: Baseline vs LLM Comparison
# =============================================================================
print("\n[2/3] Creating Baseline Comparison figure...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Flesch-Kincaid Grade Level
ax1 = axes[0]
baseline_fk = [advanced_results['baseline_comparison_by_role'][role]['baseline_fk_mean'] 
              for role in roles]
llm_fk = [advanced_results['baseline_comparison_by_role'][role]['llm_fk_mean'] 
         for role in roles]

x = np.arange(len(roles))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_fk, width, label='Raw SHAP Baseline', 
               color='#888888', edgecolor='black')
bars2 = ax1.bar(x + width/2, llm_fk, width, label='Adaptive LLM', 
               color=colors, edgecolor='black')

ax1.set_ylabel('Flesch-Kincaid Grade Level', fontsize=12)
ax1.set_xlabel('Clinician Role', fontsize=12)
ax1.set_title('Reading Difficulty: Baseline vs Adaptive LLM\n(Lower = Easier for Patients)', 
             fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(role_labels, rotation=15)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add reference lines
ax1.axhline(y=8, color='orange', linestyle=':', alpha=0.7, label='Middle School')
ax1.axhline(y=12, color='red', linestyle=':', alpha=0.7, label='High School')

# Right plot: Reading Ease
ax2 = axes[1]
baseline_re = [advanced_results['baseline_comparison_by_role'][role]['baseline_re_mean'] 
              for role in roles]
llm_re = [advanced_results['baseline_comparison_by_role'][role]['llm_re_mean'] 
         for role in roles]

bars3 = ax2.bar(x - width/2, baseline_re, width, label='Raw SHAP Baseline', 
               color='#888888', edgecolor='black')
bars4 = ax2.bar(x + width/2, llm_re, width, label='Adaptive LLM', 
               color=colors, edgecolor='black')

ax2.set_ylabel('Flesch Reading Ease Score', fontsize=12)
ax2.set_xlabel('Clinician Role', fontsize=12)
ax2.set_title('Reading Ease: Baseline vs Adaptive LLM\n(Higher = Easier to Read)', 
             fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(role_labels, rotation=15)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 110)

# Add value labels for patient improvement
patient_improvement = llm_re[-1] - baseline_re[-1]
ax2.annotate(f'+{patient_improvement:.1f}', 
            xy=(x[-1] + width/2, llm_re[-1]),
            xytext=(x[-1] + 0.5, llm_re[-1] + 5),
            fontsize=11, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure15_baseline_comparison.png'), dpi=150, bbox_inches='tight')
print(f"   Saved: figure15_baseline_comparison.png")
plt.close()

# =============================================================================
# FIGURE 3: Combined Summary (Grounding + Adaptation Trade-off)
# =============================================================================
print("\n[3/3] Creating Trade-off Analysis figure...")

fig, ax = plt.subplots(figsize=(10, 8))

# X-axis: Reading Ease (simplicity)
# Y-axis: Grounding Score (faithfulness)
# Size: Word count

reading_ease = [advanced_results['baseline_comparison_by_role'][role]['llm_re_mean'] 
               for role in roles]
grounding = [advanced_results['grounding_by_role'][role]['mean_grounding'] * 100 
            for role in roles]

# Get word counts from original metrics
with open(os.path.join(RESULTS_DIR, 'aggregated_metrics.json'), 'r') as f:
    original_metrics = json.load(f)
word_counts = [original_metrics[role]['avg_word_count'] for role in roles]

# Normalize sizes for visualization
sizes = [w * 8 for w in word_counts]

scatter = ax.scatter(reading_ease, grounding, s=sizes, c=colors, 
                    edgecolors='black', linewidth=2, alpha=0.8)

# Add labels
for i, role in enumerate(role_labels):
    ax.annotate(f'{role}\n({word_counts[i]:.0f} words)', 
               (reading_ease[i], grounding[i]),
               xytext=(10, 10), textcoords='offset points',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add quadrant labels
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

ax.text(85, 90, 'IDEAL:\nSimple & Faithful', fontsize=10, ha='center', 
       style='italic', color='green', fontweight='bold')
ax.text(15, 90, 'Technical &\nFaithful', fontsize=10, ha='center', 
       style='italic', color='blue')
ax.text(85, 15, 'Simple but\nLess Explicit', fontsize=10, ha='center', 
       style='italic', color='orange')

ax.set_xlabel('Reading Ease Score (Higher = Simpler)', fontsize=12)
ax.set_ylabel('Feature Grounding Score (%) (Higher = More Faithful)', fontsize=12)
ax.set_title('Simplicity vs Faithfulness Trade-off Analysis\n(Size = Word Count)', 
            fontsize=14, fontweight='bold')
ax.set_xlim(0, 110)
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure16_tradeoff.png'), dpi=150, bbox_inches='tight')
print(f"   Saved: figure16_tradeoff.png")
plt.close()

print("\n" + "=" * 60)
print("✅ ADVANCED VISUALIZATIONS COMPLETE!")
print("=" * 60)
print("\nCreated 3 new figures:")
print("   • figure14_grounding.png - LLM faithfulness to SHAP features")
print("   • figure15_baseline_comparison.png - Raw SHAP vs Adaptive LLM")
print("   • figure16_tradeoff.png - Simplicity vs Faithfulness trade-off")
print("\n📝 These figures demonstrate NOVEL contributions for your paper!")
