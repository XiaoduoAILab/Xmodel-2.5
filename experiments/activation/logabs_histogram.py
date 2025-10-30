#!/usr/bin/env python3
"""
log10_weight_dist.py
统计任意 PyTorch 模型所有参数 |w| 的 log10 分布。

用法:
    python log10_weight_dist.py <model_name_or_path> [--bins 50] [--range -7 0]
"""

import argparse
import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def parse_args():
    parser = argparse.ArgumentParser(description="统计模型参数 |w| 的 log10 分布")
    parser.add_argument("model", help="HuggingFace 模型名称或本地路径")
    parser.add_argument("--bins", type=int, default=50, help="直方图 bins 数量")
    parser.add_argument("--range", type=float, nargs=2, default=[-7.0, 0.0],
                        metavar=("LOW", "HIGH"), help="log10(|w|) 的统计范围")
    parser.add_argument("--no-plot", action="store_true",
                        help="不画图，仅打印统计信息")
    parser.add_argument("--device", default="cpu",
                        help="加载模型的设备，如 cpu / cuda / auto")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载模型
    print(f"Loading model: {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device,
        trust_remote_code=True
    )
    print("Model loaded.")

    # 2. 收集所有 >0 参数
    all_params = []
    for _, p in model.named_parameters():
        mask = p.data.abs() > 0
        if mask.any():
            all_params.append(p.data[mask].abs())
    if not all_params:
        raise ValueError("No valid parameters found!")
    weights_abs = torch.cat(all_params)

    # 3. 计算 log10(|w|)  —— 加 .float() 消除 BFloat16
    log10_abs = torch.log10(weights_abs.float())

    # 4. 直方图统计
    hist, edges = torch.histogram(log10_abs,
                                  bins=args.bins,
                                  range=tuple(args.range))
    hist = hist.cpu().numpy()
    bins = 0.5 * (edges[:-1] + edges[1:]).cpu().numpy()

    # 5. 打印摘要
    print("\n=== 统计摘要 ===")
    print(f"总参数(>0)数量: {len(weights_abs)}")
    print(f"log10(|w|) 均值: {log10_abs.mean().item():.3f}")
    print(f"log10(|w|) 中位数: {log10_abs.median().item():.3f}")
    print(f"log10(|w|) 最大值: {log10_abs.max().item():.3f}")
    print(f"log10(|w|) 最小值: {log10_abs.min().item():.3f}")

    # 6. 拟合正态分布  ------------------ NEW ------------------
    mu, std = norm.fit(log10_abs.cpu().numpy())
    print(f"\n=== 正态分布拟合 ===")
    print(f"均值 μ ≈ {mu:.4f}")
    print(f"标准差 σ ≈ {std:.4f}")

    # 7. 画图（叠加拟合曲线）
    if not args.no_plot:
        plt.figure(figsize=(7, 4))
        # 画直方图
        plt.bar(bins, hist, width=bins[1]-bins[0],
                color="steelblue", label="Histogram")

        # 画拟合曲线
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 300)
        p = norm.pdf(x, mu, std) * (bins[1]-bins[0]) * len(log10_abs)
        plt.plot(x, p, 'r', linewidth=2, label=f'Normal fit μ={mu:.3f}, σ={std:.3f}')

        plt.xlabel(r"$\log_{10}(|\theta|)$")
        plt.ylabel("Counts")
        plt.title(f"Histogram & Normal fit for {args.model}")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()