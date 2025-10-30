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
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    # # 绘制decoupled_lr与lm loss的关系
    # plt.figure(figsize=(12, 6))
    # # 绘制所有点（蓝色）
    # sns.scatterplot(data=df, x='params_decoupled_lr', y='value', color='blue', alpha=0.6)
    # # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_decoupled_lr', y='value', color='red')
    # plt.title('decoupled_lr vs lm-loss (Top 10 lowest in red)')
    # plt.xlabel('decoupled_lr')
    # plt.ylabel('lm loss')
    # plt.xscale('log')
    # plt.grid(True)
    # plt.show()

    # 绘制init_std与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_init_std', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_init_std', y='value', color='red')
    plt.title('init_std vs lm-loss (Top 10 lowest in red)')
    plt.xlabel('init_std')
    plt.ylabel('lm loss')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    # # 绘制mup_attention_residual_scale与lm loss的关系
    # plt.figure(figsize=(12, 6))
    # # 绘制所有点（蓝色）
    # sns.scatterplot(data=df, x='params_mup_attention_residual_scale', y='value', color='blue', alpha=0.6)
    # # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_mup_attention_residual_scale', y='value', color='red')
    # plt.title('mup_attention_residual_scale vs lm-loss (Top 10 lowest in red)')
    # plt.xlabel('mup_attention_residual_scale')
    # plt.ylabel('lm loss')
    # plt.grid(True)
    # plt.show()
    #
    # # 绘制mup_ffn_residual_scale与lm loss的关系
    # plt.figure(figsize=(12, 6))
    # # 绘制所有点（蓝色）
    # sns.scatterplot(data=df, x='params_mup_ffn_residual_scale', y='value', color='blue', alpha=0.6)
    # # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_mup_ffn_residual_scale', y='value', color='red')
    # plt.title('mup_ffn_residual_scale vs lm-loss (Top 10 lowest in red)')
    # plt.xlabel('mup_ffn_residual_scale')
    # plt.ylabel('lm loss')
    # plt.grid(True)
    # plt.show()

    # 绘制输入缩放与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_global_batch_size', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_global_batch_size', y='value', color='red')
    plt.title('global_batch_size vs. lm loss (Top 10 lowest in red)')
    plt.xlabel('global_batch_size')
    plt.ylabel('lm loss')
    plt.grid(True)
    plt.show()

    # 绘制输出缩放与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_muon_matched_adamw_rms', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_muon_matched_adamw_rms', y='value', color='red')
    plt.title('muon_matched_adamw_rms vs. lm loss (Top 10 lowest in red)')
    plt.xlabel('muon_matched_adamw_rms')
    plt.ylabel('lm loss')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # file_path = "hpo_results/hpo_mup_v6_results_08101038.csv"
    # file_path = "hpo_results/hpo_mup_v8_results_08161200.csv"
    # file_path = "hpo_results/hpo_mup_v8_results_08170931.csv"
    file_path = "hpo_results/hpo_mup_v8_d_results_08180734.csv"
    # file_path = "hpo_results/hpo_mup_v8_w_results_08181117.csv"
    visualize_hpo_results(file_path)
