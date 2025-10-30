import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_hpo_results(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 检查数据
    print(df.head())

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 绘制learning rate与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 标记最小的10个点
    df['is_top10'] = df['value'].isin(df['value'].nsmallest(10))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_learning_rate', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_learning_rate', y='value', color='red')
    plt.title('learning rate vs lm-loss (Top 10 lowest in red)')
    plt.xlabel('learning rate')
    plt.ylabel('lm loss')
    # plt.ylim(4.5, 12)
    # plt.xscale('log')
    plt.grid(True)
    plt.show()

    # 绘制decoupled_lr与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_init_std', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_init_std', y='value', color='red')
    plt.title('init_std vs lm-loss (Top 10 lowest in red)')
    plt.xlabel('init_std')
    plt.ylabel('lm loss')
    # plt.ylim(5.5, 12)
    # plt.xscale('log')
    plt.grid(True)
    plt.show()

    # 绘制mup_attention_residual_scale与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mup_attention_residual_scale', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mup_attention_residual_scale', y='value', color='red')
    plt.title('mup_attention_residual_scale vs lm-loss (Top 10 lowest in red)')
    plt.xlabel('mup_attention_residual_scale')
    plt.ylabel('lm loss')
    # plt.ylim(5.5, 12)
    plt.grid(True)
    plt.show()

    # 绘制mup_ffn_residual_scale与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mup_ffn_residual_scale', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mup_ffn_residual_scale', y='value', color='red')
    plt.title('mup_ffn_residual_scale vs lm-loss (Top 10 lowest in red)')
    plt.xlabel('mup_ffn_residual_scale')
    plt.ylabel('lm loss')
    # plt.ylim(5.5, 12)
    plt.grid(True)
    plt.show()

    # 绘制输入缩放与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mup_input_scale', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mup_input_scale', y='value', color='red')
    plt.title('muP input scale vs. lm loss (Top 10 lowest in red)')
    plt.xlabel('muP input scale')
    # plt.xscale('log')
    plt.ylabel('lm loss')
    # plt.ylim(5.5, 12)
    plt.grid(True)
    plt.show()

    # 绘制输出缩放与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mup_output_scale', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mup_output_scale', y='value', color='red')
    plt.title('muP output scale vs. lm loss (Top 10 lowest in red)')
    plt.xlabel('muP output scale')
    # plt.xscale('log')
    plt.ylabel('lm loss')
    # plt.ylim(5.5, 12)
    plt.grid(True)
    plt.show()

    # # 绘制use_mup与lm loss的关系
    # plt.figure(figsize=(12, 6))
    # # 绘制所有点（蓝色）
    # sns.scatterplot(data=df, x='params_use_muon', y='value', color='blue', alpha=0.6)
    # # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_use_muon', y='value', color='red')
    # plt.title('use_muon vs. lm loss (Top 10 lowest in red)')
    # plt.xlabel('use_muon')     
    # plt.ylabel('lm loss')
    # plt.ylim(5.5, 12)
    # plt.grid(True)
    # plt.show()

    # 绘制use_depth_mup与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_use_mup', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_use_mup', y='value', color='red')
    plt.title('use_mup vs. lm loss (Top 10 lowest in red)')
    plt.xlabel('use_mup')
    plt.ylabel('lm loss')
    # plt.ylim(5.5, 12)
    plt.grid(True)
    plt.show()

    # # 绘制use_depth_mup与lm loss的关系
    # plt.figure(figsize=(12, 6))
    # # 绘制所有点（蓝色）
    # sns.scatterplot(data=df, x='params_use_fp8', y='value', color='blue', alpha=0.6)
    # # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_use_fp8', y='value', color='red')
    # plt.title('use_fp8 vs. lm loss (Top 10 lowest in red)')
    # plt.xlabel('use_fp8')
    # plt.ylabel('lm loss')
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    # file_path = "hpo_results/hpo_mup_v3_results_08021856.csv"
    # file_path = "hpo_results/mup_v17_results_09041000.csv"
    # file_path = "hpo_results/mup_v18_results_09051001.csv"
    # file_path = "hpo_results/adamw_mup_v1_results_09060949.csv"
    # file_path = "hpo_results/adamw_mup_v2_results_09062318.csv"
    # file_path = "hpo_results/adamw_mup_v3_results_09081632.csv"
    # file_path = "hpo_results/adamw_mup_v4_results_09091309.csv"
    file_path = "hpo_results/adamw_mup_v4_results_09082254.csv"
    visualize_hpo_results(file_path)
