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

    # 绘制learning rate与rms distance的关系
    plt.figure(figsize=(12, 6))
    # 标记最小的10个点
    df['is_top10'] = df['value'].isin(df['value'].nsmallest(10))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_init_std', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_init_std', y='value', color='red')
    plt.title('init std vs rms distance (Top 10 lowest in red)')
    plt.xlabel('init std')
    plt.ylabel('rms distance to 1.0')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    # 绘制mup_attention_residual_scale与rms distance的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mup_attention_residual_scale', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mup_attention_residual_scale', y='value', color='red')
    plt.title('mup_attention_residual_scale vs rms-distance (Top 10 lowest in red)')
    plt.xlabel('mup_attention_residual_scale')
    plt.ylabel('rms distance')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    # 绘制mup_ffn_residual_scale与rms distance的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mup_ffn_residual_scale', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mup_ffn_residual_scale', y='value', color='red')
    plt.title('mup_ffn_residual_scale vs rms-distance (Top 10 lowest in red)')
    plt.xlabel('mup_ffn_residual_scale')
    plt.ylabel('rms distance')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    # 绘制输入缩放与rms distance的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mup_input_scale', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mup_input_scale', y='value', color='red')
    plt.title('muP input scale vs. rms distance (Top 10 lowest in red)')
    plt.xlabel('muP input scale')
    plt.ylabel('rms distance')
    plt.grid(True)
    plt.show()

    # 绘制输出缩放与rms distance的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mup_output_scale', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mup_output_scale', y='value', color='red')
    plt.title('muP output scale vs. rms distance (Top 10 lowest in red)')
    plt.xlabel('muP output scale')
    plt.ylabel('rms distance')
    plt.grid(True)
    plt.show()

    # 绘制mup_attention_residual_scale与rms distance的关系
    # plt.figure(figsize=(12, 6))
    # # 绘制所有点（蓝色）
    # sns.scatterplot(data=df, x='params_mup_attention_residual_scale', y='value', color='blue', alpha=0.6)
    # # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_mup_attention_residual_scale', y='value', color='red')
    # plt.title('mup_attention_residual_scale vs rms-distance (Top 10 lowest in red)')
    # plt.xlabel('mup_attention_residual_scale')
    # plt.ylabel('rms distance')
    # plt.grid(True)
    # plt.show()
    #
    # # 绘制mup_ffn_residual_scale与rms distance的关系
    # plt.figure(figsize=(12, 6))
    # # 绘制所有点（蓝色）
    # sns.scatterplot(data=df, x='params_mup_ffn_residual_scale', y='value', color='blue', alpha=0.6)
    # # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_mup_ffn_residual_scale', y='value', color='red')
    # plt.title('mup_ffn_residual_scale vs rms-distance (Top 10 lowest in red)')
    # plt.xlabel('mup_ffn_residual_scale')
    # plt.ylabel('rms distance')
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    # file_path = "hpo_results/rms1_search_results_09040839.csv"
    file_path = "hpo_results/optuna_rms1_search_results_09061201.csv"
    visualize_hpo_results(file_path)


