import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_time_comparison(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Filter for completed trials
    df = df[df['state'] == 'COMPLETE']
    
    # Convert pandas timedelta string to hours
    def convert_timedelta_to_hours(td_str):
        try:
            parts = td_str.split()
            time_part = parts[-1] if len(parts) > 1 else parts[0]
            h, m, s = map(float, time_part.split(':'))
            return h + m/60 + s/3600
        except (ValueError, AttributeError, IndexError):
            return None
    
    df['duration_hours'] = df['duration'].apply(convert_timedelta_to_hours)
    df = df.dropna(subset=['duration_hours'])  # Remove rows with invalid durations
    print(f"Analyzing {len(df)} valid trials with duration data")
    
    # Filter for the layer depths we want to plot
    num_layers = [4, 8, 16, 32, 64]
    df = df[df['params_num_layers'].isin(num_layers)]
    
    # Set up plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar plot
    ax = sns.barplot(
        data=df,
        x='params_num_layers',
        y='duration_hours',
        hue='params_use_muon',
        palette='viridis',
        estimator='mean',
        errorbar=None
    )
    
    # Customize plot
    plt.title('Average Training Time by Layer Depth (AdamW vs Muon)')
    plt.xlabel('Number of Layers')
    plt.ylabel('Training Time (hours)')
    plt.legend(title='Optimizer', labels=['AdamW', 'Muon'])
    
    # Save and show
    plt.tight_layout()
    plt.savefig('hpo_results/muon_time_comparison.png')
    plt.show()

if __name__ == "__main__":
    file_path = "hpo_results/hpo_muon_results_07310815.csv"
    visualize_time_comparison(file_path)
