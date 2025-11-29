<h1 align="center">
Xmodel-2.5: 1.3B Data-Efficient Reasoning SLM
</h1>

<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Xiaoduo%20HuggingFace-blue.svg)](https://huggingface.co/XiaoduoAILab/Xmodel-2.5)
[![arXiv](https://img.shields.io/badge/Arxiv-2511.19496-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.19496) 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/XiaoduoAILab/Xmodel-2.5/blob/main/LICENSE)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/XiaoduoAILab/Xmodel-2.5)
[![github](https://img.shields.io/github/stars/XiaoduoAILab/Xmodel-2.5.svg?style=social)](https://github.com/XiaoduoAILab/Xmodel-2.5)  

</h5>

<p align="center">
  <a href="README.md">中文版</a> | <a href="README.en.md">English Version</a>
</p>

## ✨ Features

- **1.3B parameter small language model** designed for complex reasoning tasks
- **Extreme data efficiency**: Trained with only 1.4T tokens, achieving performance close to larger models
- **Three-stage WSD training strategy**: Warmup-Stable-Decay curriculum learning
- **Muon optimizer switching**: Switching from AdamW to Muon during decay phase, improving reasoning performance by 4.58%
- **FP8 mixed precision training**: 30% training throughput improvement with no precision loss
- **16K long context support**: Achieved through lightweight context extension
- **Fully open source**: Code, recipes, and evaluation tools all open

## 🌟 Introduction

Xmodel-2.5 is a 1.3 billion parameter small language model specifically designed as a **lightweight agent core**. The model builds upon Xmodel-2 with four key upgrades:

1. **Full μP support**: Extended Megatron-LM to support maximal update parameterization, enabling reliable hyperparameter transfer
2. **Efficient tokenizer**: Adopted 129K token DeepSeek-v3 tokenizer, improving compression rate and decoding speed  
3. **FP8 mixed precision**: Using E4M3 forward and E5M2 backward FP8 formats, balancing precision and throughput
4. **Optimizer scheduling**: Switching from AdamW to Muon during decay phase, significantly improving downstream task performance

Trained with only 1.4T tokens, Xmodel-2.5 achieves **52.49%** average accuracy on 13 reasoning benchmarks, ranking second among 1-2B parameter models, only behind Qwen3 (56.96%), but with 25.7x fewer training tokens.

## 📊 Benchmark

### Comprehensive Reasoning Performance

| Model | Parameters | Training Tokens | 13-Task Average |
|-------|------------|-----------------|------------------|
| Qwen3-1.7B | 1.7B | 36T | 56.96% |
| **Xmodel-2.5** | **1.3B** | **1.4T** | **52.49%** |
| InternLM2.5-1.8B | 1.8B | - | 50.19% |
| Xmodel-2-1.2B | 1.2B | 1.5T | 50.34% |
| MiniCPM-1B | 1B | - | 48.95% |
| SmolLM2-1.7B | 1.7B | 11T | 46.88% |
| Llama-3.2-1B | 1B | 9T | 44.72% |

### Detailed Task Performance

| Task | Xmodel-2.5 | Xmodel-2 | Improvement |
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

1. Clone the repository and enter the directory
   ```bash
   git clone https://github.com/XiaoduoAILab/Xmodel-2.5.git
   cd Xmodel-2.5
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## 🗝️ Quick Start

#### Download Xmodel-2.5 Model

The model files are fully open-sourced on HuggingFace and can be downloaded [here](https://huggingface.co/XiaoduoAILab/Xmodel-2.5). We provide both pre-trained model and instruction-tuned versions.

#### Xmodel-2.5 Inference Example

After downloading the model files, you can run the following script for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_path = os.path.expanduser("/path/to/Xmodel-2.5")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

prompt = "Explain the concept of transfer learning in machine learning."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generation configuration
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

output = tokenizer.decode(
    generated_ids[0][len(model_inputs.input_ids[0]):], 
    skip_special_tokens=True
)
print("Generated Response:")
print(output)
```

## 🏗️ Training Details

### Model Architecture

| Hyperparameter | Value |
|--------|-----|
| Hidden size | 1536 |
| Intermediate size | 3840 |
| Transformer layers | 48 |
| Attention heads (Q) | 24 |
| KV heads (GQA) | 8 |
| Sequence length | 3712 |
| Max position embeddings | 131072 |
| RoPE base | 500000 |

### Training Strategy

- **Three-stage WSD curriculum**: 560k steps, 1.4T tokens
- **Warmup phase**: 2k steps, linear learning rate increase
- **Stable phase**: 530k steps, gradually increasing batch size
- **Decay phase**: 20k steps, mixing 66.9% high-quality SFT data
- **Long context adaptation**: 10k additional steps, supporting 16K context

### Technical Innovations

- **μP hyperparameter transfer**: Direct transfer from 20M parameter proxy model to full model
- **Optimizer switching**: AdamW→Muon during decay phase, improving reasoning performance
- **FP8 mixed precision**: FP8 format significantly improves training efficiency

## 📜 Citation

If Xmodel-2.5 is helpful for your research or applications, please consider citing our work:

```bibtex
@misc{liu2025xmodel25,
      title={Xmodel-2.5: 1.3B Data-Efficient Reasoning SLM}, 
      author={Yang Liu and Xiaolong Zhong and Ling Jiang},
      year={2025},
      eprint={2511.19496},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.19496}, 
}
```

## 📞 Contact

If you have questions or suggestions, please contact us through:
- GitHub Issues: [Xmodel-2.5 Issues](https://github.com/XiaoduoAILab/Xmodel-2.5/issues)
- Email: foamilu@yeah.net

## 📄 License

This project is licensed under Apache-2.0. See the [LICENSE](LICENSE) file for details.
