import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import timedelta


def parse_duration(duration_str):
    """Convert duration string to total seconds"""
    parts = duration_str.split()
    days = int(parts[0])
    time_parts = parts[2].split(':')
    hours, minutes, seconds = map(float, time_parts)
    total_seconds = (days * 86400) + (hours * 3600) + (minutes * 60) + seconds
    return total_seconds


def visualize_hpo_results(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Convert duration to seconds
    df['duration_seconds'] = df['duration'].apply(parse_duration)

    # Set plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create boxplot comparing abs_activation=True vs False
    ax = sns.boxplot(
        x='params_use_abs_activation',
        y='duration_seconds',
        data=df,
        palette=['#1f77b4', '#ff7f0e']
    )

    # Add labels and title
    ax.set(
        xlabel='Use Absolute Activation',
        ylabel='Execution Time (seconds)',
        title='Execution Time Comparison: abs_activation=True vs False'
    )
    ax.set_xticklabels(['False', 'True'])

    # Calculate and print mean execution times
    mean_true = df[df['params_use_abs_activation'] == True]['duration_seconds'].mean()
    mean_false = df[df['params_use_abs_activation'] == False]['duration_seconds'].mean()
    print(f"\nMean execution time (abs_activation=True): {mean_true:.2f} seconds")
    print(f"Mean execution time (abs_activation=False): {mean_false:.2f} seconds")
    print(f"Difference: {abs(mean_true - mean_false):.2f} seconds")

    # Add boxplot explanation
    plt.figtext(0.5, -0.1,
        "Boxplot Explanation:\n"
        "- The box shows the interquartile range (IQR, middle 50% of data)\n"
        "- The line in the box is the median\n"
        "- Whiskers show 1.5*IQR range\n"
        "- Points beyond whiskers are outliers",
        ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('abs_activation_time_comparison.png', bbox_inches='tight')
    print("Saved execution time plot to abs_activation_time_comparison.png")

    # Create new figure for lm loss comparison
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        x='params_use_abs_activation',
        y='value',
        data=df,
        palette=['#1f77b4', '#ff7f0e']
    )
    ax.set(
        xlabel='Use Absolute Activation',
        ylabel='LM Loss',
        title='LM Loss Comparison: abs_activation=True vs False'
    )
    ax.set_xticklabels(['False', 'True'])

    # Calculate and print mean lm loss
    mean_true = df[df['params_use_abs_activation'] == True]['value'].mean()
    mean_false = df[df['params_use_abs_activation'] == False]['value'].mean()
    print(f"\nMean LM loss (abs_activation=True): {mean_true:.4f}")
    print(f"Mean LM loss (abs_activation=False): {mean_false:.4f}")
    print(f"Difference: {abs(mean_true - mean_false):.4f}")

    plt.tight_layout()
    plt.savefig('abs_activation_loss_comparison.png', bbox_inches='tight')
    print("Saved LM loss plot to abs_activation_loss_comparison.png")

    # Create depth comparison plot
    plt.figure(figsize=(15, 10))
    layers = [4, 8, 16, 32, 64]
    
    for i, num_layer in enumerate(layers, 1):
        plt.subplot(2, 3, i)
        layer_data = df[df['params_num_layers'] == num_layer]
        
        # Plot True and False cases
        for use_abs, color in [(True, '#ff7f0e'), (False, '#1f77b4')]:
            subset = layer_data[layer_data['params_use_abs_activation'] == use_abs]
            plt.scatter(subset['params_learning_rate'], subset['value'],
                      color=color, label=f'abs_activation={use_abs}', alpha=0.7)
        
        plt.title(f'Num Layers: {num_layer}')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('LM Loss')
        plt.xscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()

    plt.suptitle('LM Loss Comparison by Layer Depth (abs_activation=True vs False)', y=1.02)
    plt.tight_layout()
    plt.savefig('depth_comparison_plot.png', bbox_inches='tight', dpi=300)
    print("Saved depth comparison plot to depth_comparison_plot.png")


if __name__ == "__main__":
    file_path = "hpo_results/hpo_mup_d_abs_results_07300748.csv"
    visualize_hpo_results(file_path)
