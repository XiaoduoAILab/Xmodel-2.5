import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def visualize_hpo_results(file_path):
    """Visualize HPO results with learning rate vs loss for different batch sizes.
    
    Args:
        file_path (str): Path to CSV file containing HPO results
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Get unique batch sizes
    batch_sizes = [6, 12, 24, 48, 96, 192, 240, 384, 480, 960]
    
    plt.figure(figsize=(12, 8))
    sns.set(style='whitegrid')
    
    # Plot each batch size
    for bs in batch_sizes:
        subset = df[df['params_global_batch_size'] == bs]
        if not subset.empty:
            # Sort by learning rate before plotting
            subset = subset.sort_values('params_learning_rate')
            plt.semilogx(subset['params_learning_rate'], subset['value'], 
                        label=f'BS={bs}', marker='o')
            
            # Mark point with minimum loss in red
            min_loss_idx = subset['value'].idxmin()
            plt.semilogx(subset.loc[min_loss_idx, 'params_learning_rate'], 
                        subset.loc[min_loss_idx, 'value'], 
                        'ro', markersize=8)
    
    plt.xlabel('Learning Rate (log scale)', fontsize=12)
    plt.ylabel('LM Loss', fontsize=12)
    plt.title('LM Loss vs Learning Rate by Batch Size', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join('hpo_plots', 'lr_vs_loss_by_bs.png')
    os.makedirs('hpo_plots', exist_ok=True)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    # file_path = "hpo_results/hpo_mup_w_v9_hs64_results_08110838.csv"
    # file_path = "hpo_results/hpo_mup_w_v9_hs128_results_08110836.csv"
    # file_path = "hpo_results/hpo_mup_w_v9_hs256_results_08110832.csv"
    # file_path = "hpo_results/hpo_mup_w_v10_hs64_results_08121736.csv"
    file_path = "hpo_results/hpo_mup_w_v10_hs128_results_08130826.csv"
    
    visualize_hpo_results(file_path)
