import glob
import torch
import os
import csv


def get_param_mean_std(ckpt_path):
    # 获得检查点参数的均值和标准差
    state_dict = torch.load(f'{ckpt_path}/pytorch_model.bin', map_location="cpu")
    
    # 收集所有参数值
    all_params = []
    for param_name, param_tensor in state_dict.items():
        # 跳过非参数张量（如buffers）
        if param_tensor.is_floating_point():
            all_params.append(param_tensor.view(-1))
    
    if not all_params:
        return 0.0, 0.0
    
    # 合并所有参数
    all_params = torch.cat(all_params)
    
    # 计算均值和标准差
    mean = all_params.mean().item()
    std = all_params.std().item()

    return mean, std

if __name__ == "__main__":
    folders = glob.glob("/data2/liuyang/pretrain_xmodel_i_line/out/*-hf")
    folders = sorted(folders)
    
    # 收集所有结果
    all_results = {}
    iteration_names = []
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        ckpts = glob.glob(f"{folder}/iter_00*")
        ckpts = sorted(ckpts)
        
        mean_list, std_list = [], []
        for ckpt in ckpts:
            ckpt_name = os.path.basename(ckpt)
            if not iteration_names:
                iteration_names = [os.path.basename(ckpt) for ckpt in ckpts]
            mean, std = get_param_mean_std(ckpt)
            mean_list.append(mean)
            std_list.append(std)
            print(f"{folder}\t{ckpt_name}\t{mean:.6f}\t{std:.6f}")
        
        all_results[folder_name + "_mean"] = mean_list
        all_results[folder_name + "_std"] = std_list
    
    # 写入CSV文件
    with open('param_statistics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入列名
        headers = ['iteration'] + list(all_results.keys())
        writer.writerow(headers)
        
        # 写入数据行（5行对应5个迭代）
        for i in range(len(iteration_names)):
            row = [iteration_names[i]]
            for key in all_results.keys():
                row.append(all_results[key][i])
            writer.writerow(row)
    
    print("Results saved to param_statistics.csv")
