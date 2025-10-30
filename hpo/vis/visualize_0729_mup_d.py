import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_hpo_results(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Filter out infinite values only
    df = df[df['value'] != float('inf')]

    # Filter for the 5 layer depths we want to plot
    num_layers = [4, 8, 16, 32, 64]
    df = df[df['params_num_layers'].isin(num_layers)]

    # Set up plot
    sns.set(style="whitegrid", palette="viridis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Learning Rate vs Training Loss (SP vs μP) by Layer Depth')

    # Plot SP (params_use_depth_mup=False)
    sp_df = df[df['params_use_depth_mup'] == False]
    for layers in num_layers:
        layers_df = sp_df[sp_df['params_num_layers'] == layers]
        layers_df = layers_df.sort_values('params_learning_rate')
        ax1.plot(layers_df['params_learning_rate'], layers_df['value'],
                 label=f'{layers}', marker='o')
        # Mark minimum point in red
        min_idx = layers_df['value'].idxmin()
        ax1.plot(layers_df.loc[min_idx, 'params_learning_rate'],
                 layers_df.loc[min_idx, 'value'], 'ro')
    ax1.set_xscale('log')
    ax1.set_xticks([2 ** n for n in range(-15, -2)])
    ax1.set_xticklabels([f'2^{n}' for n in range(-15, -2)])
    ax1.set_xlim(2 ** -15, 2 ** -3)
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Training Loss')
    ax1.set_ylim(3.5, 7.5)
    ax1.set_title('Standard Parameterization (SP)')
    ax1.legend(title='Num Layers')

    # Plot μP (params_use_depth_mup=True)
    mup_df = df[df['params_use_depth_mup'] == True]
    for layers in num_layers:
        layers_df = mup_df[mup_df['params_num_layers'] == layers]
        layers_df = layers_df.sort_values('params_learning_rate')
        ax2.plot(layers_df['params_learning_rate'], layers_df['value'],
                 label=f'{layers}', marker='o')
        # Mark minimum point in red
        min_idx = layers_df['value'].idxmin()
        ax2.plot(layers_df.loc[min_idx, 'params_learning_rate'],
                 layers_df.loc[min_idx, 'value'], 'ro')
    ax2.set_xscale('log')
    ax2.set_xticks([2 ** n for n in range(-15, -2)])
    ax2.set_xticklabels([f'2^{n}' for n in range(-15, -2)])
    ax2.set_xlim(2 ** -15, 2 ** -3)
    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Training Loss')
    ax2.set_ylim(3.5, 7.5)
    ax2.set_title('Maximal Update Parameterization (μP)')
    ax2.legend(title='Num Layers')

    plt.tight_layout()
    plt.savefig('hpo_results/depth_comparison_plot.png')
    plt.show()


if __name__ == "__main__":
    # file_path = "hpo_results/hpo_results_mup_d_202507290832.csv"
    # file_path = "hpo_results/hpo_mup_d_results_07300822.csv"
    # file_path = "hpo_results/hpo_mup_d_results_08010902.csv"
    # file_path = "hpo_results/hpo_mup_d_v4_results_08010908.csv"
    # file_path = "hpo_results/hpo_mup_d_v5_results_08021205.csv"
    file_path = "hpo_results/hpo_mup_d_v6_results_08030937.csv"
    visualize_hpo_results(file_path)
