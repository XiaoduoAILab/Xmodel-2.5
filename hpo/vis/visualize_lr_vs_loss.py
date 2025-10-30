import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_time_comparison(file_path):
    # Read and filter data
    df = pd.read_csv(file_path)
    df = df[df['value'] != float('inf')]
    
    # Convert duration string to seconds
    def parse_duration(duration_str):
        parts = duration_str.split()
        days = int(parts[0])
        time_parts = parts[2].split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    
    df['duration_sec'] = df['duration'].apply(parse_duration)
    
    # Set style and palette
    sns.set_style("whitegrid")
    palette = sns.color_palette("viridis", 3)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot training time vs num_layers for each decay mode
    for decay_mode in [0, 1, 2]:
        subset = df[df['params_decay_mode'] == decay_mode]
        if len(subset) > 0:
            # Group by num_layers and calculate mean time
            time_data = subset.groupby('params_num_layers')['duration_sec'].mean().reset_index()
            
            # Plot line
            sns.lineplot(
                x='params_num_layers',
                y='duration_sec',
                data=time_data,
                color=palette[decay_mode],
                label=f'decay_mode={decay_mode}',
                marker='o'
            )
    
    plt.title('Training Time Comparison by Decay Mode')
    plt.xlabel('Number of Layers')
    plt.ylabel('Training Time (seconds)')
    plt.legend()
    plt.savefig('training_time_comparison.png', bbox_inches='tight')
    plt.close()


def plot_lr_vs_loss(file_path):
    # Read and filter data
    df = pd.read_csv(file_path)
    df = df[df['value'] != float('inf')]
    
    # Set style and palette
    sns.set_style("whitegrid")
    palette = sns.color_palette("viridis", 3)
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 20))
    fig.tight_layout(pad=5.0)
    
    # Plot for each num_layers value
    num_layers = [4, 8, 16, 32, 64]
    for i, n in enumerate(num_layers):
        ax = axes[i]
        ax.set_title(f'Training Loss vs Learning Rate (num_layers={n})')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Training Loss')
        ax.set_xscale('log')
        
        # Plot each decay mode
        for decay_mode in [0, 1, 2]:
            subset = df[(df['params_num_layers'] == n) & 
                       (df['params_decay_mode'] == decay_mode)]
            if len(subset) > 0:
                # Sort by learning rate for proper line plotting
                subset = subset.sort_values('params_learning_rate')
                
                # Plot line
                line = sns.lineplot(
                    x='params_learning_rate',
                    y='value',
                    data=subset,
                    ax=ax,
                    color=palette[decay_mode],
                    label=f'decay_mode={decay_mode}'
                )
                
                # Mark minimum point
                min_idx = subset['value'].idxmin()
                min_point = subset.loc[min_idx]
                ax.scatter(
                    min_point['params_learning_rate'],
                    min_point['value'],
                    color='red',
                    s=100,
                    zorder=5
                )
        
        ax.legend()
    
    # Save figure
    plt.savefig('training_loss_vs_lr_comparison.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    file_path = "hpo_results/hpo_wd_decouple_results_07311629.csv"
    plot_lr_vs_loss(file_path)
    plot_time_comparison(file_path)
