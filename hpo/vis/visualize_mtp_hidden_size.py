import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def visualize_mtp_hidden_size_results(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Filter out infinite values
    df = df[df['value'] != float('inf')]
    
    # Define the hidden sizes and mtp layers to plot
    hidden_sizes = [256, 512, 1024, 2048]
    mtp_layers = [0, 1, 2, 3]
    
    # Set up the plot style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12
    
    # Create 4 subplots (one for each hidden size)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define colors for different mtp layers
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, hidden_size in enumerate(hidden_sizes):
        ax = axes[i]
        
        # Filter data for current hidden size
        size_df = df[df['params_hidden_size'] == hidden_size]
        
        # Plot each mtp layer
        for j, mtp_layer in enumerate(mtp_layers):
            # Filter data for current mtp layer
            layer_df = size_df[size_df['params_mtp_num_layers'] == mtp_layer]
            
            if not layer_df.empty:
                # Sort by learning rate
                layer_df = layer_df.sort_values('params_learning_rate')
                
                # Plot the line
                ax.plot(layer_df['params_learning_rate'], layer_df['value'],
                       label=f'MTP Layers={mtp_layer}', 
                       color=colors[j], 
                       marker=markers[j],
                       markersize=6,
                       linewidth=2)
                
                # Mark minimum point
                min_idx = layer_df['value'].idxmin()
                ax.plot(layer_df.loc[min_idx, 'params_learning_rate'],
                       layer_df.loc[min_idx, 'value'], 
                       'ko', markersize=8)
        
        # Set plot properties
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('LM Loss')
        ax.set_title(f'Hidden Size = {hidden_size}')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Set consistent x and y limits for better comparison
        ax.set_xlim(2**-15, 2**-2)
        ax.set_ylim(3, 9)
    
    plt.tight_layout()
    plt.savefig('hpo_results/mtp_hidden_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual plots for each hidden size
    for hidden_size in hidden_sizes:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter data for current hidden size
        size_df = df[df['params_hidden_size'] == hidden_size]
        
        # Plot each mtp layer
        for j, mtp_layer in enumerate(mtp_layers):
            # Filter data for current mtp layer
            layer_df = size_df[size_df['params_mtp_num_layers'] == mtp_layer]
            
            if not layer_df.empty:
                # Sort by learning rate
                layer_df = layer_df.sort_values('params_learning_rate')
                
                # Plot the line
                ax.plot(layer_df['params_learning_rate'], layer_df['value'],
                       label=f'MTP Layers={mtp_layer}', 
                       color=colors[j], 
                       marker=markers[j],
                       markersize=6,
                       linewidth=2)
                
                # Mark minimum point
                min_idx = layer_df['value'].idxmin()
                ax.plot(layer_df.loc[min_idx, 'params_learning_rate'],
                       layer_df.loc[min_idx, 'value'], 
                       'ko', markersize=8)
        
        # Set plot properties
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('LM Loss')
        ax.set_title(f'Hidden Size = {hidden_size}')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.set_xlim(2**-15, 2**-2)
        ax.set_ylim(3, 9)
        
        plt.tight_layout()
        plt.savefig(f'hpo_results/mtp_hidden_size_{hidden_size}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    file_path = "hpo_results/mtp_v1_results_08242010.csv"
    visualize_mtp_hidden_size_results(file_path)
