import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM

model_path = "/data2/liuyang/models/Xmodel2.5/random_init"  # 替换为你的模型路径
# 假设 model 已经实例化好并且随机初始化
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model.eval()

# 用来存放三组参数
param_groups = defaultdict(list)

for name, param in model.named_parameters():
    if param.requires_grad:                   # 只统计可训练参数
        if name.startswith("model.embed_tokens"):
            param_groups["embed_tokens"].append(param.detach().flatten())
        elif name.startswith("lm_head"):
            param_groups["lm_head"].append(param.detach().flatten())
        else:
            param_groups["others"].append(param.detach().flatten())

# 把每组展平成一个大向量
embed_vec   = torch.cat(param_groups["embed_tokens"])
# lm_head_vec = torch.cat(param_groups["lm_head"])
others_vec  = torch.cat(param_groups["others"])

# 计算标准差
std = {
    "embed_tokens.std": embed_vec.std().item(),
    # "lm_head.std":      lm_head_vec.std().item(),
    "others.std":       others_vec.std().item(),
}

print(std)