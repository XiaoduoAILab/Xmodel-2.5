import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def visualize_three_plots(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Filter out infinite values only
    df = df[df['value'] != float('inf')]

    # Filter for the 4 hidden sizes we want to plot
    hidden_sizes = [256, 512, 1024, 2048]
    df = df[df['params_hidden_size'].isin(hidden_sizes)]

    # Set up the plot style
    sns.set(style="whitegrid", palette="viridis")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    # Create 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MLA vs GQA Comparison Across Different Metrics', fontsize=16)

    # Define colors and markers for different hidden sizes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    # Define line styles for MLA vs GQA
    line_styles = {'MLA': '-', 'GQA': '--'}

    # Plot 1: Learning Rate vs LM Loss
    for i, size in enumerate(hidden_sizes):
        for use_mla, style in [(True, 'MLA'), (False, 'GQA')]:
            size_df = df[(df['params_hidden_size'] == size) & (df['params_use_mla'] == use_mla)]
            size_df = size_df.sort_values('params_learning_rate')
            if not size_df.empty:
                ax1.plot(size_df['params_learning_rate'], size_df['value'],
                        label=f'{size} {style}', 
                        color=colors[i], 
                        marker=markers[i], 
                        linestyle=line_styles[style],
                        markersize=6,
                        linewidth=2)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('LM Loss')
    ax1.set_title('(a) Learning Rate vs LM Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning Rate vs Model Size (Billions of Parameters)
    for i, size in enumerate(hidden_sizes):
        for use_mla, style in [(True, 'MLA'), (False, 'GQA')]:
            size_df = df[(df['params_hidden_size'] == size) & (df['params_use_mla'] == use_mla)]
            size_df = size_df.sort_values('params_learning_rate')
            if not size_df.empty:
                # Get the first valid parameter count for this configuration
                param_count = size_df['user_attrs_total_number_of_parameters_in_billions'].iloc[0]
                ax2.plot(size_df['params_learning_rate'], [param_count] * len(size_df),
                        label=f'{size} {style}', 
                        color=colors[i], 
                        marker=markers[i], 
                        linestyle=line_styles[style],
                        markersize=6,
                        linewidth=2)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Model Size (Billions of Parameters)')
    ax2.set_title('(b) Learning Rate vs Model Size')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning Rate vs Training Speed (Elapsed Time per Iteration)
    for i, size in enumerate(hidden_sizes):
        for use_mla, style in [(True, 'MLA'), (False, 'GQA')]:
            size_df = df[(df['params_hidden_size'] == size) & (df['params_use_mla'] == use_mla)]
            size_df = size_df.sort_values('params_learning_rate')
            if not size_df.empty:
                ax3.plot(size_df['params_learning_rate'], size_df['user_attrs_elapsed_time_per_iteration'],
                        label=f'{size} {style}', 
                        color=colors[i], 
                        marker=markers[i], 
                        linestyle=line_styles[style],
                        markersize=6,
                        linewidth=2)
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Learning Rate (log scale)')
    ax3.set_ylabel('Training Speed (Elapsed Time per Iteration)')
    ax3.set_title('(c) Learning Rate vs Training Speed')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hpo_results/three_plots_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print some statistics for verification
    print("Data Statistics:")
    print(f"Total valid data points: {len(df)}")
    print("\nHidden size distribution:")
    print(df['params_hidden_size'].value_counts().sort_index())
    print("\nMLA vs GQA distribution:")
    print(df['params_use_mla'].value_counts())
    print("\nLearning rate range:")
    print(f"Min: {df['params_learning_rate'].min()}, Max: {df['params_learning_rate'].max()}")

if __name__ == "__main__":
    # file_path = "hpo_results/mla_v2_results_08220822.csv"
    # file_path = "hpo_results/mla_v3_results_08231039.csv"
    file_path = "hpo_results/mla_v4_results_08240111.csv"
    visualize_three_plots(file_path)
