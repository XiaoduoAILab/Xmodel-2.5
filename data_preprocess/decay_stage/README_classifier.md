# 文本分类脚本使用说明

## 功能概述

本脚本使用Qwen3-4B模型对JSONL格式的文本数据进行分类，按照两个维度：
- **主题分类**：代码（code）、数学（math）、知识（knowledge）
- **质量评分**：1-5分（5为最高质量）

## 环境要求

- Python 3.7+
- 依赖包：`requests`, `tqdm`, `multiprocessing`
- 6个Qwen3-4B模型服务运行在8000-8005端口

## 安装依赖

```bash
pip install requests tqdm
```

## 使用方法

### 基本用法

```bash
cd data_preprocess/decay_stage
python text_classifier.py
```

### 自定义参数

```bash
# 指定输入文件
python text_classifier.py --input_path /path/to/your/data.jsonl

# 指定输出文件
python text_classifier.py --output_path /path/to/output.jsonl

# 使用不同数量的进程
python text_classifier.py --num_processes 4

# 测试模式（只处理前100个样本）
python text_classifier.py --sample_size 100
```

### 完整参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_path` | `/data2/liuyang/Xmodel-2.5/sft_mixed_v2_deduped_v4.jsonl` | 输入JSONL文件路径 |
| `--output_path` | `sft_mixed_v2_deduped_v4_classified.jsonl` | 输出JSONL文件路径 |
| `--num_processes` | 6 | 并行处理进程数 |
| `--sample_size` | None | 测试样本大小（仅处理前N个样本） |

## 输出格式

分类后的JSONL文件每行包含原始数据加上分类结果：

```json
{
  "text": "原始文本内容...",
  "topic": "code|math|knowledge",
  "quality": 1|2|3|4|5
}
```

## 分类标准

### 主题分类

- **代码（code）**：包含编程代码、算法实现、技术文档等
- **数学（math）**：包含数学公式、定理证明、数学问题等  
- **知识（knowledge）**：包含百科知识、科学原理、历史人文等

### 质量评分

- **5分**：内容完整、逻辑清晰、表达准确、信息丰富
- **4分**：内容较完整、逻辑较清晰、表达较准确
- **3分**：内容基本完整、逻辑基本清晰、表达基本准确
- **2分**：内容不完整、逻辑混乱、表达不准确
- **1分**：内容严重缺失、逻辑严重混乱、表达严重不准确

## 模型服务配置

脚本假设有6个Qwen3-4B模型服务运行在以下端口：
- localhost:8000
- localhost:8001  
- localhost:8002
- localhost:8003
- localhost:8004
- localhost:8005

每个进程会循环使用这些端口进行负载均衡。

## 错误处理

- 网络请求失败会自动重试3次
- JSON解析失败会尝试从文本中提取分类结果
- 无效的分类结果会自动修正为默认值
- 空文本会自动分类为质量1分的unknown主题

## 性能优化

- 使用多进程并行处理，充分利用6个模型服务
- 每个请求设置30秒超时
- 文本长度限制为4000字符
- 使用指数退避策略处理网络错误

## 日志输出

脚本会输出详细的处理进度和分类结果统计：

```
2025-10-31 10:45:23 - INFO - Loaded 1000 items from JSONL file
2025-10-31 10:45:23 - INFO - Starting parallel classification with 6 processes
2025-10-31 10:45:23 - INFO - Split into 6 batches, each with ~167 items
2025-10-31 10:50:15 - INFO - Topic distribution:
2025-10-31 10:50:15 - INFO -   code: 350 (35.00%)
2025-10-31 10:50:15 - INFO -   knowledge: 450 (45.00%)
2025-10-31 10:50:15 - INFO -   math: 200 (20.00%)
2025-10-31 10:50:15 - INFO - Quality distribution:
2025-10-31 10:50:15 - INFO -   Quality 1: 50 (5.00%)
2025-10-31 10:50:15 - INFO -   Quality 2: 100 (10.00%)
2025-10-31 10:50:15 - INFO -   Quality 3: 300 (30.00%)
2025-10-31 10:50:15 - INFO -   Quality 4: 400 (40.00%)
2025-10-31 10:50:15 - INFO -   Quality 5: 150 (15.00%)
```

## 注意事项

1. 确保所有模型服务正常运行且可访问
2. 大文件处理可能需要较长时间，建议先使用小样本测试
3. 输出文件会覆盖已存在的同名文件
4. 分类结果仅供参考，建议人工抽样验证准确性
