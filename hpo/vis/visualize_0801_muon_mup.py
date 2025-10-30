
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def visualize_hpo_results(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Filter out infinite values
    df = df[df['value'] != float('inf')]
    
    # Viridis color palette
    palette = sns.color_palette("viridis", n_colors=10)
    
    # Filter for num_layers=64 only
    df = df[df['params_num_layers'] == 64]
    
    # Create separate plots for each muon_matched_adamw_rms
    for muon_rms in [0.1, 0.2, 0.3, 0.4, 0.5]:
        plt.figure(figsize=(8, 6))
        
        # Plot 2 lines (use_depth_mup True/False)
        for use_depth in [True, False]:
            line_data = df[(df['params_muon_matched_adamw_rms'] == muon_rms) & 
                         (df['params_use_depth_mup'] == use_depth)]
            
            if not line_data.empty:
                # Sort by learning_rate before plotting
                line_data = line_data.sort_values('params_learning_rate')
                plt.plot(line_data['params_learning_rate'], line_data['value'],
                        color='blue' if use_depth else 'orange',
                        label=f"depth={use_depth}",
                        linewidth=2)
                
                # Mark minimum point with red dot
                min_idx = line_data['value'].idxmin()
                plt.scatter(line_data.loc[min_idx, 'params_learning_rate'], 
                          line_data.loc[min_idx, 'value'],
                          color='red', zorder=5)
        
        # Set log scale for x-axis with range 2^-15 to 2^-3
        plt.xscale('log', base=2)
        plt.xticks([2**n for n in range(-15, -2)], 
                  [f"$2^{{{n}}}$" for n in range(-15, -2)])
        plt.xlim(2**-15, 2**-3)
        
        plt.title(f"muon_rms={muon_rms} (num_layers=64)")
        plt.xlabel("Learning Rate")
        plt.ylabel("Train Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save each plot separately
        plt.savefig(f"hpo_results/muon_rms_{muon_rms}_comparison.png")
        plt.show()


if __name__ == "__main__":
    file_path = "hpo_results/hpo_muon_mup_results_08010905.csv"
    visualize_hpo_results(file_path)
