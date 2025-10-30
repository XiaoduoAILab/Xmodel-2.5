import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_hpo_results(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Filter out infinite values only
    df = df[df['value'] != float('inf')]

    # Filter for the hidden sizes available in the data
    available_hidden_sizes = sorted(df['params_hidden_size'].unique())
    df = df[df['params_hidden_size'].isin(available_hidden_sizes)]

    # Get the unique hidden sizes for plotting
    hidden_sizes = sorted(df['params_hidden_size'].unique())

    # Set up plot
    sns.set(style="whitegrid", palette="viridis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Learning Rate vs Loss (SP vs μP)')

    # Plot SP (params_use_mup=False)
    sp_df = df[df['params_use_mup'] == False]
    for size in hidden_sizes:
        size_df = sp_df[sp_df['params_hidden_size'] == size]
        size_df = size_df.sort_values('params_learning_rate')
        ax1.plot(size_df['params_learning_rate'], size_df['value'],
                 label=f'{size}', marker='o')
        # Mark minimum point in red
        min_idx = size_df['value'].idxmin()
        ax1.plot(size_df.loc[min_idx, 'params_learning_rate'],
                 size_df.loc[min_idx, 'value'], 'ro')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([2 ** n for n in range(-13, -3)])
    ax1.set_xticklabels([f'2^{n}' for n in range(-13, -3)])
    ax1.set_xlim(2 ** -13, 2 ** -3)
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(4.5, 8)
    ax1.set_title('Standard Parameterization (SP)')
    ax1.legend(title='Hidden Size')

    # Plot μP (params_use_mup=True)
    mup_df = df[df['params_use_mup'] == True]
    for size in hidden_sizes:
        size_df = mup_df[mup_df['params_hidden_size'] == size]
        size_df = size_df.sort_values('params_learning_rate')
        ax2.plot(size_df['params_learning_rate'], size_df['value'],
                 label=f'{size}', marker='o')
        # Mark minimum point in red
        min_idx = size_df['value'].idxmin()
        ax2.plot(size_df.loc[min_idx, 'params_learning_rate'],
                 size_df.loc[min_idx, 'value'], 'ro')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([2 ** n for n in range(-13, -3)])
    ax2.set_xticklabels([f'2^{n}' for n in range(-13, -3)])
    ax2.set_xlim(2 ** -13, 2 ** -3)
    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(4.5, 8)
    ax2.set_title('Maximal Update Parameterization (μP)')
    ax2.legend(title='Hidden Size')

    plt.tight_layout()
    plt.savefig('hpo_results/comparison_plot.png')
    plt.savefig('hpo_results/lr_mutransfer_width_stability.pdf')
    plt.show()


if __name__ == "__main__":
    file_path = "hpo_results/muon_mup_width_transfer_v16_results_09110026.csv"
    visualize_hpo_results(file_path)
