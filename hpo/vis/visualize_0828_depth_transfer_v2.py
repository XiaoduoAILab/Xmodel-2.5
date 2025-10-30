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
    sp_df = df[df['params_use_mup'] == False]
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
    ax1.set_xticks([2 ** n for n in range(-15, -3)])
    ax1.set_xticklabels([f'2^{n}' for n in range(-15, -3)])
    # ax2.set_xlim(2 ** -10, 2 ** -2)
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Training Loss')
    # ax2.set_ylim(4.5, 8.5)
    ax1.set_title('Standard Parameterization (SP)')
    ax1.legend(title='Num Layers')

    # Plot μP (params_use_depth_mup=True)
    mup_df = df[df['params_use_mup'] == True]
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
    ax2.set_xticks([2 ** n for n in range(-15, -3)])
    ax2.set_xticklabels([f'2^{n}' for n in range(-15, -3)])
    # ax2.set_xlim(2 ** -10, 2 ** -2)
    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Training Loss')
    # ax2.set_ylim(5.5, 11.5)
    ax2.set_title('Maximal Update Parameterization (μP)')
    ax2.legend(title='Num Layers')

    plt.tight_layout()
    plt.savefig('hpo_results/depth_comparison_plot.png')
    plt.show()


if __name__ == "__main__":
    # file_path = "hpo_results/muon_mup_depth_transfer_v1_results_08281533.csv"
    # file_path = "hpo_results/muon_mup_depth_transfer_v3_results_08290823.csv"
    # file_path = "hpo_results/muon_mup_depth_transfer_v5_results_083118088.csv"
    # file_path = "hpo_results/muon_mup_depth_transfer_v6_results_09010015.csv"
    # file_path = "hpo_results/muon_mup_depth_transfer_v7_results_09020816.csv"
    # file_path = "hpo_results/muon_mup_depth_transfer_v8_results_09021913.csv"
    # file_path = "hpo_results/muon_mup_depth_transfer_v9_results_09030753.csv"
    # file_path = "hpo_results/muon_mup_depth_transfer_v10_results_09031321.csv"
    # file_path = "hpo_results//adamw_mup_depth_transfer_v1_results_09052314.csv"
    file_path = "hpo_results/muon_mup_depth_transfer_v13_results_09100806.csv"
    visualize_hpo_results(file_path)
