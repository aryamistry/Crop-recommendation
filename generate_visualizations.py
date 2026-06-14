"""
Generate all visualization figures for the Crop Recommendation System
Useful for report generation and documentation

Author: Principal ML Engineer
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create reports directory
Path('reports/figures').mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VISUALIZATION GENERATION SCRIPT")
print("=" * 80)
print()


def generate_class_distribution(df):
    """Generate class distribution visualization."""
    print("Generating: Class Distribution...")
    
    class_counts = df['label'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(class_counts)), class_counts.values, 
                   color='skyblue', edgecolor='black', alpha=0.8)
    
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Crop Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Crop Class Distribution', fontsize=16, fontweight='bold')
    ax.axhline(y=class_counts.mean(), color='red', linestyle='--', 
               label=f'Mean: {class_counts.mean():.0f}', linewidth=2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('reports/figures/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: reports/figures/class_distribution.png")


def generate_feature_distributions(df):
    """Generate feature distribution plots."""
    print("Generating: Feature Distributions...")
    
    feature_cols = [col for col in df.columns if col != 'label']
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col], bins=50, color='steelblue', 
                      edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontweight='bold', fontsize=11)
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics
        mean = df[col].mean()
        median = df[col].median()
        axes[idx].axvline(mean, color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {mean:.2f}')
        axes[idx].axvline(median, color='green', linestyle='-.', linewidth=2, 
                         label=f'Median: {median:.2f}')
        axes[idx].legend(fontsize=8)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.suptitle('Feature Distributions', fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: reports/figures/feature_distributions.png")


def generate_correlation_matrix(df):
    """Generate correlation matrix heatmap."""
    print("Generating: Correlation Matrix...")
    
    feature_cols = [col for col in df.columns if col != 'label']
    correlation_matrix = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        vmin=-1,
        vmax=1
    )
    
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: reports/figures/correlation_matrix.png")


def generate_boxplots(df):
    """Generate box plots for outlier detection."""
    print("Generating: Outlier Detection (Box Plots)...")
    
    feature_cols = [col for col in df.columns if col != 'label']
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols):
        bp = axes[idx].boxplot(df[col], vert=True, patch_artist=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(color='blue', linewidth=1.5),
                               capprops=dict(color='blue', linewidth=1.5),
                               flierprops=dict(marker='o', markerfacecolor='red', 
                                             markersize=5, alpha=0.5))
        
        axes[idx].set_title(f'{col} - Outliers', fontweight='bold', fontsize=11)
        axes[idx].set_ylabel(col, fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        
        axes[idx].text(0.5, 0.95, f'Outliers: {outliers}',
                      transform=axes[idx].transAxes,
                      ha='center', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      fontsize=9)
    
    fig.delaxes(axes[-1])
    
    plt.suptitle('Outlier Detection (Box Plots)', fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('reports/figures/outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: reports/figures/outlier_detection.png")


def generate_feature_by_crop(df):
    """Generate feature distributions by crop type."""
    print("Generating: Feature Distributions by Crop...")
    
    feature_cols = ['N', 'P', 'K', 'rainfall']  # Top 4 features
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols):
        # Create violin plot
        crop_data = [df[df['label'] == crop][col].values 
                     for crop in sorted(df['label'].unique())]
        
        parts = axes[idx].violinplot(crop_data, positions=range(len(crop_data)),
                                      showmeans=True, showmedians=True)
        
        axes[idx].set_xticks(range(len(sorted(df['label'].unique()))))
        axes[idx].set_xticklabels(sorted(df['label'].unique()), 
                                   rotation=45, ha='right', fontsize=8)
        axes[idx].set_ylabel(col, fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{col} Distribution by Crop', 
                           fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Distributions by Crop Type', 
                 fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_by_crop.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: reports/figures/feature_by_crop.png")


def generate_summary_statistics_table(df):
    """Generate and save summary statistics table."""
    print("Generating: Summary Statistics Table...")
    
    feature_cols = [col for col in df.columns if col != 'label']
    
    stats = df[feature_cols].describe().T
    stats['missing'] = df[feature_cols].isnull().sum()
    stats['skewness'] = df[feature_cols].skew()
    stats['kurtosis'] = df[feature_cols].kurtosis()
    
    # Round for readability
    stats = stats.round(2)
    
    # Save to CSV
    stats.to_csv('reports/summary_statistics.csv')
    
    print("  ✓ Saved: reports/summary_statistics.csv")


def main():
    """Generate all visualizations."""
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        df = pd.read_csv('Crop-recommendation.csv')
        print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns\n")
    except FileNotFoundError:
        print("✗ Error: Crop-recommendation.csv not found")
        print("  Please ensure the dataset is in the project root directory")
        return
    
    # Generate all visualizations
    print("Generating visualizations...\n")
    
    try:
        generate_class_distribution(df)
        generate_feature_distributions(df)
        generate_correlation_matrix(df)
        generate_boxplots(df)
        generate_feature_by_crop(df)
        generate_summary_statistics_table(df)
        
        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("=" * 80)
        print("\nGenerated files:")
        print("  • reports/figures/class_distribution.png")
        print("  • reports/figures/feature_distributions.png")
        print("  • reports/figures/correlation_matrix.png")
        print("  • reports/figures/outlier_detection.png")
        print("  • reports/figures/feature_by_crop.png")
        print("  • reports/summary_statistics.csv")
        print("\n✓ All visualizations are ready for your report!")
        
    except Exception as e:
        print(f"\n✗ Error generating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
