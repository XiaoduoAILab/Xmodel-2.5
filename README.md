<h1 align="center">
Xmodel-2.5: 1.3B Data-Efficient Reasoning SLM
</h1>

<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Xiaoduo%20HuggingFace-blue.svg)](https://huggingface.co/XiaoduoAILab/Xmodel-2.5)
[![arXiv](https://img.shields.io/badge/Arxiv-2502.XXXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.19496) 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/XiaoduoAILab/Xmodel-2.5/blob/main/LICENSE)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/XiaoduoAILab/Xmodel-2.5)
[![github](https://img.shields.io/github/stars/XiaoduoAILab/Xmodel-2.5.svg?style=social)](https://github.com/XiaoduoAILab/Xmodel-2.5)  

</h5>

## ✨ Features

- **1.3B参数小语言模型**，专为复杂推理任务设计
- **极致数据效率**：仅用1.4T tokens训练，达到接近更大模型的性能
- **三阶段WSD训练策略**：Warmup-Stable-Decay课程学习
- **Muon优化器切换**：在衰减阶段从AdamW切换到Muon，提升推理性能4.58%
- **FPS混合精度训练**：提升30%训练吞吐量，无精度损失
- **16K长上下文支持**：通过轻量级上下文扩展实现
- **完全开源**：代码、配方和评估工具全部开放

## 🌟 Introduction

Xmodel-2.5是一个13亿参数的小语言模型，专门设计作为**轻量级智能体核心**。模型在Xmodel-2的基础上进行了四项关键升级：

1. **完整μP支持**：扩展Megatron-LM以支持最大更新参数化，实现超参数可靠传递
2. **高效分词器**：采用129K token的DeepSeek-v3分词器，提升压缩率和解码速度  
3. **FPS混合精度**：使用E4M3前向和E5M2反向的FP8格式，平衡精度和吞吐量
4. **优化器调度**：在衰减阶段从AdamW切换到Muon，显著提升下游任务性能

仅用1.4T tokens训练，Xmodel-2.5在13个推理基准测试中达到**52.49%**的平均准确率，在1-2B参数模型中排名第二，仅落后于Qwen3（56.96%），但训练token数量减少25.7倍。

## 📊 Benchmark

### 综合推理性能

| Model | Parameters | Training Tokens | 13-Task Average |
|-------|------------|-----------------|------------------|
| Qwen3-1.7B | 1.7B | 36T | 56.96% |
| **Xmodel-2.5** | **1.3B** | **1.4T** | **52.49%** |
| InternLM2.5-1.8B | 1.8B | - | 50.19% |
| Xmodel-2-1.2B | 1.2B | 1.5T | 50.34% |
| MiniCPM-1B | 1B | - | 48.95% |
| SmolLM2-1.7B | 1.7B | - | 46.88% |
| Llama-3.2-1B | 1B | - | 44.72% |

### 详细任务表现

| 任务 | Xmodel-2.5 | Xmodel-2 | 提升 |
|------|------------|----------|------|
| ARC-Challenge | 48.89 | 46.16 | +2.73 |
| ARC-Easy | 76.94 | 76.22 | +0.72 |
| PIQA | 75.95 | 75.14 | +0.81 |
| HellaSwag | 67.24 | 64.05 | +3.19 |
| WinoGrande | 64.64 | 64.25 | +0.39 |
| BBH | 54.58 | 48.90 | +5.68 |
| MMLU | 51.81 | 49.98 | +1.83 |
| GSM8k | 58.98 | 56.56 | +2.42 |
| MATH | 28.94 | 25.64 | +3.30 |
| HumanEval | 28.66 | 29.27 | -0.61 |
| MBPP | 33.00 | 30.80 | +2.20 |
| CMMLU | 47.16 | 44.29 | +2.87 |
| C-Eval | 45.54 | 43.16 | +2.38 |

## 🛠️ Install

1. 克隆仓库并进入目录
   ```bash
   git clone https://github.com/XiaoduoAILab/Xmodel-2.5.git
   cd Xmodel-2.5