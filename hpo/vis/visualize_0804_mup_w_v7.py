import os

import matplotlib.pyplot as plt
import pandas as pd


def visualize_hpo_results(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    # Check data
    print(df.head())

    # Filter out infinite values
    df = df[df['value'] != float('inf')]

    # Map decay modes to simpler labels
    decay_mode_map = {
        '--decay-all': 'decay_all',
        '--no-decay-norm-bias-embed': 'no_decay_norm_bias_embed',
        '--no-decay-norm-bias': 'no_decay_norm_bias'
    }
    df['decay_mode'] = df['params_decay_mode'].map(decay_mode_map)

    # Create output directory if needed
    os.makedirs('hpo_plots', exist_ok=True)

    # Generate plots for each optimizer and decay mode combination
    for optimizer in ['adamw', 'muon']:
        for decay_mode in decay_mode_map.values():
            # Filter data for current combination
            subset = df[(df['params_optimizer'] == optimizer) &
                        (df['decay_mode'] == decay_mode)]
            print(f"Processing optimizer: {optimizer}, decay mode: {decay_mode}")
            print(f"Subset size: {subset.shape[0]}")

            # Check data
            print(subset.head())

            if subset.empty:
                continue

            plt.figure(figsize=(10, 6))

            # Plot each hidden_size
            for hidden_size in [64, 128, 256, 512, 1024, 2048]:
                # Filter for current hidden_size
                data = subset[subset['params_hidden_size'] == hidden_size]
                if data.empty:
                    continue

                # Sort by learning rate
                data = data.sort_values('params_learning_rate')

                # Plot line
                line, = plt.plot(data['params_learning_rate'], data['value'],
                                 label=f'hidden_size={hidden_size}')

                # Find and mark minimum point
                min_idx = data['value'].idxmin()
                plt.scatter(data.loc[min_idx, 'params_learning_rate'],
                            data.loc[min_idx, 'value'],
                            color='red', zorder=5)

            plt.xscale('log')
            plt.xlabel('Learning Rate (log scale)')
            plt.ylabel('LM Loss')
            plt.title(f'Optimizer: {optimizer}, Decay Mode: {decay_mode}')
            plt.legend()
            plt.grid(True, which="both", ls="-")

            # Save plot
            filename = f'hpo_plots/{optimizer}_{decay_mode}.png'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    # file_path = "hpo_results/hpo_mup_w_v7_results_08041352.csv"
    # file_path = "hpo_results/hpo_mup_w_v7_results_08061659.csv"
    # file_path = "hpo_results/hpo_mup_w_v8_results_08070821.csv"
    file_path = "hpo_results/hpo_mup_w_v8_results_08071514.csv"
    visualize_hpo_results(file_path)
