import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

# 1. 模型路径
# model_dir = "/data2/liuyang/pretrain_xmodel_i_line/out/mup_convert_debug_output_logit_fwd-hf/iter_0008000"
model_dir = "/data2/liuyang/i_line_exp/i_line_s1-hf/iter_0080000/"

# 2. 加载模型（只加载权重，不加载 optimizer 状态）
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True
)

# 3. 收集所有可训练参数（并转 float32）
all_params = torch.cat([p.detach().float().view(-1) for p in model.parameters() if p.requires_grad])

# 4. 画图
plt.figure(figsize=(6, 3))
sns.histplot(all_params.numpy(), bins=300, kde=False, stat="density", color="steelblue")

# 5. 标出 FP8-E4M3 甜区
plt.axvspan(-1, 1, color="gold", alpha=0.15, label="FP8-E4M3 sweet spot (|x|≤1)")
plt.axvline(0, color="black", linewidth=0.5)

plt.title("Parameter distribution of muP-trained HF model")
plt.xlabel("Parameter value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# 6. (可选) 打印几个统计量
print(f"Total params: {len(all_params):,}")
print(f"Within ±1: {(all_params.abs() <= 1).float().mean().item()*100:.2f}%")