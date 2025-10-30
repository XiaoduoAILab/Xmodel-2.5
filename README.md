# Xmodel-2.5

[English Version](#english-version) | [中文版本](#中文版本)

---

## 中文版本

# Xmodel-2.5: 面向轻量级智能体的1.3B参数推理模型

<div align="center">

![Xmodel-2.5](images/model_table.png)

**一个专为边缘设备和成本敏感场景设计的1.3B参数小型语言模型**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/XiaoduoAILab/Xmodel-2.5)](https://github.com/XiaoduoAILab/Xmodel-2.5/stargazers)

</div>

## 📖 概述

Xmodel-2.5 是一个1.3B参数的解码器专用语言模型，专门为轻量级智能体应用设计。相比传统7-13B参数的大型语言模型，Xmodel-2.5在保持强大推理能力的同时，显著降低了计算资源需求，使其成为边缘设备和成本敏感部署场景的理想选择。

### 🚀 主要特性

- **高效架构**: 1.3B参数，48层，1536隐藏维度，支持131K上下文长度
- **先进训练技术**: 采用最大更新参数化(μP)、FP8混合精度训练和Warmup-Stable-Decay课程学习
- **强大推理能力**: 在1-2B参数模型中，在复杂推理和交互式任务上达到最优性能
- **多语言支持**: 支持中英文，在中文理解任务上表现优异
- **开源许可**: Apache 2.0许可证，完全开源

## 🏗️ 模型架构

Xmodel-2.5采用深度-瘦身解码器专用架构：

| 参数 | 值 |
|------|-----|
| 参数量 | 1.3B |
| 隐藏层维度 | 1536 |
| 层数 | 48 |
| 注意力头数 | 24 |
| KV头数(GQA) | 8 |
| 前馈网络维度 | 3840 |
| 最大序列长度 | 3712 |
| 位置编码 | RoPE (base=500000) |
| 词汇表大小 | 129,280 (DeepSeek-v3分词器) |

## 📊 性能表现

### 推理任务表现

| 任务 | Xmodel-2.5 | 同规模最佳模型 |
|------|-------------|----------------|
| ARC-Challenge | 46.16% | +0.66% |
| MMLU | 49.98% | +1.23% |
| GSM8K | 56.56% | +14.56% |
| C-Eval | 43.16% | +8.59% |
| CMMLU | 44.29% | +9.77% |

### 交互式智能体任务

| 任务 | 成功率 |
|------|---------|
| HotpotQA | 13.7% |
| FEVER | 40.0% |
| AlfWorld | 7.8% |
| WebShop | 2.2% |

## 🛠️ 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- Megatron-LM

### 安装

```bash
# 克隆仓库
git clone https://github.com/XiaoduoAILab/Xmodel-2.5.git
cd Xmodel-2.5

# 安装依赖
pip install -r requirements.txt
```

### 训练示例

```bash
# 运行基线训练脚本
cd examples/xmodel
bash baseline.sh
```

### 使用Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.configuration_xmodel2 import XmodelConfig
from models.modeling_xmodel2 import XmodelForCausalLM

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("XiaoduoAILab/Xmodel-2.5")
model = AutoModelForCausalLM.from_pretrained("XiaoduoAILab/Xmodel-2.5")

# 推理示例
input_text = "中国的首都是"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## 📁 项目结构

```
Xmodel-2.5/
├── models/                    # 模型定义
│   ├── configuration_xmodel2.py
│   └── modeling_xmodel2.py
├── examples/                  # 使用示例
│   └── xmodel/
│       ├── baseline.sh
│       ├── fp8.sh
│       └── data_config.json
├── data_preprocess/           # 数据预处理脚本
├── hpo/                       # 超参数优化
├── tools/                     # 工具脚本
├── docs/                      # 文档
└── pretrain_gpt.py           # 主训练脚本
```

## 🔧 训练配置

### 训练阶段

1. **预热阶段(Warmup)**: 2k步，学习率线性上升
2. **稳定阶段(Stable)**: 530k步，批量大小逐步增加
3. **衰减阶段(Decay)**: 40k步，混合高质量SFT数据

### 数据组成

- **稳定阶段**: 多样化预训练数据，包含13个数据源
- **衰减阶段**: 高质量教学数据，包含10个数据源，63.88%为SFT混合数据

## 📄 引用

如果您在研究中使用了Xmodel-2.5，请引用我们的论文：

```bibtex
@article{liu2025xmodel,
  title={Xmodel-2.5: 1B-scale Agentic Reasoner with Stable and Efficient Pre-training},
  author={Liu, Yang and Zhong, Xiaolong and Jiang, Ling},
  journal={arXiv preprint},
  year={2025}
}
```

## 🤝 贡献

我们欢迎社区贡献！请查看[贡献指南](CONTRIBUTING.md)了解如何参与。

## 📜 许可证

本项目采用Apache 2.0许可证 - 详见[LICENSE](LICENSE)文件。

## 📞 联系我们

- 邮箱: {zhongxiaolong,liuyangfoam}@xiaoduotech.com
- GitHub Issues: [问题反馈](https://github.com/XiaoduoAILab/Xmodel-2.5/issues)

---

## English Version

# Xmodel-2.5: 1.3B Parameter Reasoning Model for Lightweight Agents

<div align="center">

![Xmodel-2.5](images/model_table.png)

**A 1.3B parameter small language model designed for edge devices and cost-sensitive scenarios**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/XiaoduoAILab/Xmodel-2.5)](https://github.com/XiaoduoAILab/Xmodel-2.5/stargazers)

</div>

## 📖 Overview

Xmodel-2.5 is a 1.3B parameter decoder-only language model specifically designed for lightweight agent applications. Compared to traditional 7-13B parameter large language models, Xmodel-2.5 maintains strong reasoning capabilities while significantly reducing computational requirements, making it an ideal choice for edge devices and cost-sensitive deployment scenarios.

### 🚀 Key Features

- **Efficient Architecture**: 1.3B parameters, 48 layers, 1536 hidden dimensions, supports 131K context length
- **Advanced Training Techniques**: Utilizes Maximal Update Parameterization (μP), FP8 mixed-precision training, and Warmup-Stable-Decay curriculum learning
- **Strong Reasoning Capabilities**: Achieves state-of-the-art performance in complex reasoning and interactive tasks among 1-2B parameter models
- **Multilingual Support**: Supports Chinese and English, with excellent performance on Chinese understanding tasks
- **Open Source License**: Apache 2.0 license, fully open source

## 🏗️ Model Architecture

Xmodel-2.5 adopts a deep-and-thin decoder-only architecture:

| Parameter | Value |
|-----------|-------|
| Parameters | 1.3B |
| Hidden Size | 1536 |
| Layers | 48 |
| Attention Heads | 24 |
| KV Heads (GQA) | 8 |
| FFN Dimension | 3840 |
| Max Sequence Length | 3712 |
| Position Encoding | RoPE (base=500000) |
| Vocabulary Size | 129,280 (DeepSeek-v3 tokenizer) |

## 📊 Performance

### Reasoning Task Performance

| Task | Xmodel-2.5 | Best in Class (1-2B) |
|------|-------------|----------------------|
| ARC-Challenge | 46.16% | +0.66% |
| MMLU | 49.98% | +1.23% |
| GSM8K | 56.56% | +14.56% |
| C-Eval | 43.16% | +8.59% |
| CMMLU | 44.29% | +9.77% |

### Interactive Agent Tasks

| Task | Success Rate |
|------|--------------|
| HotpotQA | 13.7% |
| FEVER | 40.0% |
| AlfWorld | 7.8% |
| WebShop | 2.2% |

## 🛠️ Quick Start

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- Megatron-LM

### Installation

```bash
# Clone repository
git clone https://github.com/XiaoduoAILab/Xmodel-2.5.git
cd Xmodel-2.5

# Install dependencies
pip install -r requirements.txt
```

### Training Example

```bash
# Run baseline training script
cd examples/xmodel
bash baseline.sh
```

### Using Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.configuration_xmodel2 import XmodelConfig
from models.modeling_xmodel2 import XmodelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("XiaoduoAILab/Xmodel-2.5")
model = AutoModelForCausalLM.from_pretrained("XiaoduoAILab/Xmodel-2.5")

# Inference example
input_text = "The capital of China is"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## 📁 Project Structure

```
Xmodel-2.5/
├── models/                    # Model definitions
│   ├── configuration_xmodel2.py
│   └── modeling_xmodel2.py
├── examples/                  # Usage examples
│   └── xmodel/
│       ├── baseline.sh
│       ├── fp8.sh
│       └── data_config.json
├── data_preprocess/           # Data preprocessing scripts
├── hpo/                       # Hyperparameter optimization
├── tools/                     # Utility scripts
├── docs/                      # Documentation
└── pretrain_gpt.py           # Main training script
```

## 🔧 Training Configuration

### Training Phases

1. **Warmup Phase**: 2k steps, linear learning rate increase
2. **Stable Phase**: 530k steps, gradually increasing batch size
3. **Decay Phase**: 40k steps, mixed with high-quality SFT data

### Data Composition

- **Stable Phase**: Diverse pre-training data from 13 sources
- **Decay Phase**: High-quality instructional data from 10 sources, 63.88% SFT mixed data

## 📄 Citation

If you use Xmodel-2.5 in your research, please cite our paper:

```bibtex
@article{liu2025xmodel,
  title={Xmodel-2.5: 1B-scale Agentic Reasoner with Stable and Efficient Pre-training},
  author={Liu, Yang and Zhong, Xiaolong and Jiang, Ling},
  journal={arXiv preprint},
  year={2025}
}
```

## 🤝 Contributing

We welcome community contributions! Please check the [Contributing Guidelines](CONTRIBUTING.md) for details on how to participate.

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- Email: {zhongxiaolong,liuyangfoam}@xiaoduotech.com
- GitHub Issues: [Issue Tracker](https://github.com/XiaoduoAILab/Xmodel-2.5/issues)
