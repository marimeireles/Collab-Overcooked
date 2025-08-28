"""
Analysis of correlation between success rates and hierarchy types.
Creates visualizations to explore the relationship between hierarchy structure
and task success in multi-agent scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_merge_data():
    """Load and merge the hierarchy and statistics datasets."""
    print("Loading datasets...")
    
    # Load hierarchy aggregation data
    hierarchy_df = pd.read_csv('hierarchy_aggregation.csv')
    print(f"Hierarchy data shape: {hierarchy_df.shape}")
    
    # Load statistics data
    stats_df = pd.read_csv('src/eval_result/statistics_data_final-jsons-from-symmetric-data.csv')
    print(f"Statistics data shape: {stats_df.shape}")
    
    # Merge datasets on model_combination = model and recipe = order
    merged_df = pd.merge(
        hierarchy_df, 
        stats_df, 
        left_on=['model_combination', 'recipe'], 
        right_on=['model', 'order'],
        how='inner'
    )
    
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Missing values in merged data: {merged_df.isnull().sum().sum()}")
    
    return merged_df

def prepare_hierarchy_data(df):
    """Prepare data for hierarchy analysis by creating long format."""
    # Create long format for hierarchy types
    hierarchy_data = []
    
    for idx, row in df.iterrows():
        # Add records for each hierarchy type
        if row['fuzzy_hierarchy'] > 0:
            for _ in range(int(row['fuzzy_hierarchy'])):
                hierarchy_data.append({
                    'hierarchy_type': 'Fuzzy Hierarchy',
                    'success_rate': row['success_rate'],
                    'model_combination': row['model_combination'],
                    'recipe': row['recipe'],
                    'avg_interactions': row['avg_interactions'],
                    'level': row['level']
                })
        
        if row['no_hierarchy'] > 0:
            for _ in range(int(row['no_hierarchy'])):
                hierarchy_data.append({
                    'hierarchy_type': 'No Hierarchy',
                    'success_rate': row['success_rate'],
                    'model_combination': row['model_combination'],
                    'recipe': row['recipe'],
                    'avg_interactions': row['avg_interactions'],
                    'level': row['level']
                })
        
        if row['clear_hierarchy'] > 0:
            for _ in range(int(row['clear_hierarchy'])):
                hierarchy_data.append({
                    'hierarchy_type': 'Clear Hierarchy',
                    'success_rate': row['success_rate'],
                    'model_combination': row['model_combination'],
                    'recipe': row['recipe'],
                    'avg_interactions': row['avg_interactions'],
                    'level': row['level']
                })
    
    return pd.DataFrame(hierarchy_data)

def create_violin_plot(hierarchy_long_df):
    """Create violin plot showing success rate distribution by hierarchy type."""
    plt.figure(figsize=(12, 8))
    
    # Create the violin plot
    ax = sns.violinplot(
        data=hierarchy_long_df,
        x='hierarchy_type',
        y='success_rate',
        inner='box',
        palette=['#FF6B6B', '#4ECDC4', '#45B7D1']
    )
    
    # Customize the plot
    plt.title('Success Rate Distribution by Hierarchy Type', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hierarchy Type', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    
    # Add statistical annotations
    hierarchy_types = hierarchy_long_df['hierarchy_type'].unique()
    success_rates_by_type = [
        hierarchy_long_df[hierarchy_long_df['hierarchy_type'] == ht]['success_rate']
        for ht in hierarchy_types
    ]
    
    # Add mean lines
    for i, success_rates in enumerate(success_rates_by_type):
        mean_val = success_rates.mean()
        ax.hlines(mean_val, i-0.2, i+0.2, colors='red', linestyles='solid', linewidth=2)
        ax.text(i, mean_val + 0.05, f'μ={mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('violin_plot_success_by_hierarchy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n=== VIOLIN PLOT SUMMARY STATISTICS ===")
    for hierarchy_type in hierarchy_types:
        subset = hierarchy_long_df[hierarchy_long_df['hierarchy_type'] == hierarchy_type]['success_rate']
        print(f"\n{hierarchy_type}:")
        print(f"  Count: {len(subset)}")
        print(f"  Mean: {subset.mean():.4f}")
        print(f"  Median: {subset.median():.4f}")
        print(f"  Std: {subset.std():.4f}")
        print(f"  Min: {subset.min():.4f}")
        print(f"  Max: {subset.max():.4f}")

def create_hierarchy_proportion_analysis(df):
    """Analyze the proportion of hierarchy types and their success rates."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate hierarchy proportions for each row
    df['total_trials'] = df['fuzzy_hierarchy'] + df['no_hierarchy'] + df['clear_hierarchy']
    df['fuzzy_prop'] = df['fuzzy_hierarchy'] / df['total_trials']
    df['no_prop'] = df['no_hierarchy'] / df['total_trials']
    df['clear_prop'] = df['clear_hierarchy'] / df['total_trials']
    
    # 1. Scatter plot: Success rate vs Fuzzy hierarchy proportion
    axes[0,0].scatter(df['fuzzy_prop'], df['success_rate'], alpha=0.6, color='#FF6B6B')
    axes[0,0].set_xlabel('Fuzzy Hierarchy Proportion')
    axes[0,0].set_ylabel('Success Rate')
    axes[0,0].set_title('Success Rate vs Fuzzy Hierarchy Proportion')
    
    # Add correlation
    corr_fuzzy, p_fuzzy = pearsonr(df['fuzzy_prop'], df['success_rate'])
    axes[0,0].text(0.05, 0.95, f'r = {corr_fuzzy:.3f}\np = {p_fuzzy:.3f}', 
                   transform=axes[0,0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Scatter plot: Success rate vs Clear hierarchy proportion
    axes[0,1].scatter(df['clear_prop'], df['success_rate'], alpha=0.6, color='#45B7D1')
    axes[0,1].set_xlabel('Clear Hierarchy Proportion')
    axes[0,1].set_ylabel('Success Rate')
    axes[0,1].set_title('Success Rate vs Clear Hierarchy Proportion')
    
    corr_clear, p_clear = pearsonr(df['clear_prop'], df['success_rate'])
    axes[0,1].text(0.05, 0.95, f'r = {corr_clear:.3f}\np = {p_clear:.3f}', 
                   transform=axes[0,1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Scatter plot: Success rate vs No hierarchy proportion
    axes[1,0].scatter(df['no_prop'], df['success_rate'], alpha=0.6, color='#4ECDC4')
    axes[1,0].set_xlabel('No Hierarchy Proportion')
    axes[1,0].set_ylabel('Success Rate')
    axes[1,0].set_title('Success Rate vs No Hierarchy Proportion')
    
    corr_no, p_no = pearsonr(df['no_prop'], df['success_rate'])
    axes[1,0].text(0.05, 0.95, f'r = {corr_no:.3f}\np = {p_no:.3f}', 
                   transform=axes[1,0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Heatmap of average success rates by hierarchy dominance
    df['dominant_hierarchy'] = df[['fuzzy_prop', 'no_prop', 'clear_prop']].idxmax(axis=1)
    df['dominant_hierarchy'] = df['dominant_hierarchy'].map({
        'fuzzy_prop': 'Fuzzy Dominant',
        'no_prop': 'No Hierarchy Dominant', 
        'clear_prop': 'Clear Dominant'
    })
    
    # Group by recipe and dominant hierarchy type
    heatmap_data = df.groupby(['recipe', 'dominant_hierarchy'])['success_rate'].mean().unstack(fill_value=0)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                ax=axes[1,1], cbar_kws={'label': 'Average Success Rate'})
    axes[1,1].set_title('Average Success Rate by Recipe and Dominant Hierarchy')
    axes[1,1].set_xlabel('Dominant Hierarchy Type')
    axes[1,1].set_ylabel('Recipe')
    
    plt.tight_layout()
    plt.savefig('hierarchy_proportion_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== CORRELATION ANALYSIS ===")
    print(f"Fuzzy Hierarchy Proportion vs Success Rate: r = {corr_fuzzy:.4f}, p = {p_fuzzy:.4f}")
    print(f"Clear Hierarchy Proportion vs Success Rate: r = {corr_clear:.4f}, p = {p_clear:.4f}")
    print(f"No Hierarchy Proportion vs Success Rate: r = {corr_no:.4f}, p = {p_no:.4f}")

def create_model_hierarchy_heatmap(df):
    """Create heatmap showing success rates across models and hierarchy types."""
    plt.figure(figsize=(16, 10))
    
    # Calculate weighted average success rate by hierarchy type for each model
    model_hierarchy_success = []
    
    for model in df['model_combination'].unique():
        model_data = df[df['model_combination'] == model]
        
        # Calculate weighted averages
        total_fuzzy = model_data['fuzzy_hierarchy'].sum()
        total_clear = model_data['clear_hierarchy'].sum()
        total_no = model_data['no_hierarchy'].sum()
        
        if total_fuzzy > 0:
            fuzzy_success = (model_data['fuzzy_hierarchy'] * model_data['success_rate']).sum() / total_fuzzy
        else:
            fuzzy_success = np.nan
            
        if total_clear > 0:
            clear_success = (model_data['clear_hierarchy'] * model_data['success_rate']).sum() / total_clear
        else:
            clear_success = np.nan
            
        if total_no > 0:
            no_success = (model_data['no_hierarchy'] * model_data['success_rate']).sum() / total_no
        else:
            no_success = np.nan
        
        model_hierarchy_success.append({
            'Model': model,
            'Fuzzy Hierarchy': fuzzy_success,
            'Clear Hierarchy': clear_success,
            'No Hierarchy': no_success
        })
    
    heatmap_df = pd.DataFrame(model_hierarchy_success).set_index('Model')
    
    # Create the heatmap
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Average Success Rate'})
    
    plt.title('Average Success Rate by Model and Hierarchy Type', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hierarchy Type', fontsize=14, fontweight='bold')
    plt.ylabel('Model Combination', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('model_hierarchy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interaction_analysis(df):
    """Analyze the relationship between hierarchy, interactions, and success."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Success rate vs Average interactions, colored by dominant hierarchy
    df['total_trials'] = df['fuzzy_hierarchy'] + df['no_hierarchy'] + df['clear_hierarchy']
    df['fuzzy_prop'] = df['fuzzy_hierarchy'] / df['total_trials']
    df['no_prop'] = df['no_hierarchy'] / df['total_trials']
    df['clear_prop'] = df['clear_hierarchy'] / df['total_trials']
    
    df['dominant_hierarchy'] = df[['fuzzy_prop', 'no_prop', 'clear_prop']].idxmax(axis=1)
    df['dominant_hierarchy'] = df['dominant_hierarchy'].map({
        'fuzzy_prop': 'Fuzzy',
        'no_prop': 'No Hierarchy', 
        'clear_prop': 'Clear'
    })
    
    hierarchy_colors = {'Fuzzy': '#FF6B6B', 'Clear': '#45B7D1', 'No Hierarchy': '#4ECDC4'}
    
    for hierarchy_type in df['dominant_hierarchy'].unique():
        subset = df[df['dominant_hierarchy'] == hierarchy_type]
        axes[0,0].scatter(subset['avg_interactions'], subset['success_rate'], 
                         label=hierarchy_type, alpha=0.7, color=hierarchy_colors.get(hierarchy_type, 'gray'))
    
    axes[0,0].set_xlabel('Average Interactions')
    axes[0,0].set_ylabel('Success Rate')
    axes[0,0].set_title('Success Rate vs Interactions by Dominant Hierarchy')
    axes[0,0].legend()
    
    # 2. Box plot of interactions by hierarchy type
    hierarchy_long_df = prepare_hierarchy_data(df)
    sns.boxplot(data=hierarchy_long_df, x='hierarchy_type', y='avg_interactions', ax=axes[0,1])
    axes[0,1].set_title('Interaction Distribution by Hierarchy Type')
    axes[0,1].set_xlabel('Hierarchy Type')
    axes[0,1].set_ylabel('Average Interactions')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Success rate distribution by task level
    sns.boxplot(data=df, x='level', y='success_rate', ax=axes[1,0])
    axes[1,0].set_title('Success Rate Distribution by Task Level')
    axes[1,0].set_xlabel('Task Level')
    axes[1,0].set_ylabel('Success Rate')
    
    # 4. Hierarchy type distribution by task level
    level_hierarchy_data = []
    for idx, row in df.iterrows():
        level_hierarchy_data.extend([
            {'level': row['level'], 'hierarchy_type': 'Fuzzy', 'count': row['fuzzy_hierarchy']},
            {'level': row['level'], 'hierarchy_type': 'Clear', 'count': row['clear_hierarchy']},
            {'level': row['level'], 'hierarchy_type': 'No Hierarchy', 'count': row['no_hierarchy']}
        ])
    
    level_hierarchy_df = pd.DataFrame(level_hierarchy_data)
    level_hierarchy_pivot = level_hierarchy_df.groupby(['level', 'hierarchy_type'])['count'].sum().unstack(fill_value=0)
    
    # Normalize to proportions
    level_hierarchy_prop = level_hierarchy_pivot.div(level_hierarchy_pivot.sum(axis=1), axis=0)
    
    level_hierarchy_prop.plot(kind='bar', stacked=True, ax=axes[1,1], 
                             color=[hierarchy_colors[col] for col in level_hierarchy_prop.columns])
    axes[1,1].set_title('Hierarchy Type Distribution by Task Level')
    axes[1,1].set_xlabel('Task Level')
    axes[1,1].set_ylabel('Proportion')
    axes[1,1].legend(title='Hierarchy Type')
    axes[1,1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('interaction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def statistical_analysis(df, hierarchy_long_df):
    """Perform statistical tests on hierarchy and success relationships."""
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS")
    print("="*50)
    
    # ANOVA test for hierarchy types
    fuzzy_success = hierarchy_long_df[hierarchy_long_df['hierarchy_type'] == 'Fuzzy Hierarchy']['success_rate']
    clear_success = hierarchy_long_df[hierarchy_long_df['hierarchy_type'] == 'Clear Hierarchy']['success_rate']
    no_hierarchy_success = hierarchy_long_df[hierarchy_long_df['hierarchy_type'] == 'No Hierarchy']['success_rate']
    
    # Remove empty groups
    groups = [fuzzy_success, clear_success, no_hierarchy_success]
    non_empty_groups = [group for group in groups if len(group) > 0]
    group_names = ['Fuzzy', 'Clear', 'No Hierarchy']
    non_empty_names = [name for group, name in zip(groups, group_names) if len(group) > 0]
    
    if len(non_empty_groups) > 1:
        f_stat, p_value = stats.f_oneway(*non_empty_groups)
        print(f"\nANOVA Test (Hierarchy Type vs Success Rate):")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Pairwise t-tests
    print(f"\nPairwise t-tests:")
    for i in range(len(non_empty_groups)):
        for j in range(i+1, len(non_empty_groups)):
            t_stat, p_val = stats.ttest_ind(non_empty_groups[i], non_empty_groups[j])
            print(f"{non_empty_names[i]} vs {non_empty_names[j]}: t={t_stat:.4f}, p={p_val:.4f}")
    
    # Correlation analysis with hierarchy proportions
    df['total_trials'] = df['fuzzy_hierarchy'] + df['no_hierarchy'] + df['clear_hierarchy']
    df['fuzzy_prop'] = df['fuzzy_hierarchy'] / df['total_trials']
    df['clear_prop'] = df['clear_hierarchy'] / df['total_trials']
    df['no_prop'] = df['no_hierarchy'] / df['total_trials']
    
    print(f"\nCorrelation Analysis (Hierarchy Proportions vs Success Rate):")
    
    correlations = [
        ('Fuzzy Hierarchy Proportion', df['fuzzy_prop'], df['success_rate']),
        ('Clear Hierarchy Proportion', df['clear_prop'], df['success_rate']),
        ('No Hierarchy Proportion', df['no_prop'], df['success_rate']),
        ('Average Interactions', df['avg_interactions'], df['success_rate'])
    ]
    
    for name, x, y in correlations:
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        print(f"\n{name}:")
        print(f"  Pearson: r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"  Spearman: ρ={spearman_r:.4f}, p={spearman_p:.4f}")

def main():
    """Main function to run the complete analysis."""
    print("Starting Hierarchy-Success Correlation Analysis")
    print("="*50)
    
    # Load and merge data
    df = load_and_merge_data()
    
    # Prepare hierarchy data in long format for violin plot
    hierarchy_long_df = prepare_hierarchy_data(df)
    
    print(f"\nHierarchy long format data shape: {hierarchy_long_df.shape}")
    print(f"Hierarchy types found: {hierarchy_long_df['hierarchy_type'].unique()}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Violin plot (requested)
    print("1. Creating violin plot...")
    create_violin_plot(hierarchy_long_df)
    
    # 2. Hierarchy proportion analysis
    print("2. Creating hierarchy proportion analysis...")
    create_hierarchy_proportion_analysis(df)
    
    # 3. Model-hierarchy heatmap
    print("3. Creating model-hierarchy heatmap...")
    create_model_hierarchy_heatmap(df)
    
    # 4. Interaction analysis
    print("4. Creating interaction analysis...")
    create_interaction_analysis(df)
    
    # 5. Statistical analysis
    print("5. Performing statistical analysis...")
    statistical_analysis(df, hierarchy_long_df)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("Generated files:")
    print("- violin_plot_success_by_hierarchy.png")
    print("- hierarchy_proportion_analysis.png") 
    print("- model_hierarchy_heatmap.png")
    print("- interaction_analysis.png")
    print("="*50)

if __name__ == "__main__":
    main()

