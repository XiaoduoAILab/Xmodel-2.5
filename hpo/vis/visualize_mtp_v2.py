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

    # 绘制mtp_num_layers与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mtp_num_layers', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mtp_num_layers', y='value', color='red')
    plt.title('mtp_num_layers vs lm-loss (Top 10 lowest in red)')
    plt.xlabel('mtp_num_layers')
    plt.ylabel('lm loss')
    # plt.xscale('log')
    plt.grid(True)
    plt.show()

    # 绘制mtp_scale与lm loss的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mtp_loss_scaling_factor', y='value', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    sns.scatterplot(data=df[df['is_top10']], x='params_mtp_loss_scaling_factor', y='value', color='red')
    plt.title('mtp_loss_scaling_factor vs lm-loss (Top 10 lowest in red)')
    plt.xlabel('mtp_loss_scaling_factor')
    plt.ylabel('lm loss')
    # plt.xscale('log')
    plt.grid(True)
    plt.show()

    # 绘制mtp_num_layers与elapsed_time_per_iteration的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mtp_num_layers', y='user_attrs_elapsed_time_per_iteration', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_mtp_num_layers', y='value', color='red')
    plt.title('mtp_num_layers vs. elapsed_time_per_iteration (Top 10 lowest in red)')
    plt.xlabel('mtp_num_layers')
    plt.ylabel('elapsed_time_per_iteration')
    plt.grid(True)
    plt.show()

    # 绘制mtp_num_layers与total_number_of_parameters_in_billions的关系
    plt.figure(figsize=(12, 6))
    # 绘制所有点（蓝色）
    sns.scatterplot(data=df, x='params_mtp_num_layers', y='user_attrs_total_number_of_parameters_in_billions', color='blue', alpha=0.6)
    # 高亮最小的10个点（红色）
    # sns.scatterplot(data=df[df['is_top10']], x='params_mtp_num_layers', y='value', color='red')
    plt.title('mtp_num_layers vs. total_number_of_parameters_in_billions (Top 10 lowest in red)')
    plt.xlabel('mtp_num_layers')
    plt.ylabel('total_number_of_parameters_in_billions')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = "hpo_results/mtp_v2_results_08240812.csv"
    visualize_hpo_results(file_path)
