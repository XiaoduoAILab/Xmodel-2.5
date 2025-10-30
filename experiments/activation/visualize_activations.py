import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/data2/liuyang/models/Llama-3.2-1B/',
                    help='Path to the model directory')
parser.add_argument('--n_tokens', type=int, default=10,
                    help='Number of tokens to average activations over')
args = parser.parse_args()

# 加载模型和分词器
device = torch.device('cuda:7')
model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)
config = model.config
print(model)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

# 准备输入数据
input_text = "Here is some example text for the model to process."
inputs = tokenizer(input_text, return_tensors="pt").to(device)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 初始化激活值累加器（按token位置存储）
activation_sums = {
    'word_embedding': None,
    'attention_outputs': [None for _ in range(len(model.model.layers))],
    'ffn_outputs': [None for _ in range(len(model.model.layers))],
    'output_logits': None
}

# 定义hook函数收集各层激活值
activation_data = {}


def get_activation(name):
    def hook(model, input, output):
        # Handle tuple outputs (common in transformer layers)
        if isinstance(output, tuple):
            activation_data[name] = output[0].detach().cpu().numpy()
        else:
            activation_data[name] = output.detach().cpu().numpy()

    return hook


# 注册hook
hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.self_attn.register_forward_hook(get_activation(f'attn_layer_{i}')))
    hooks.append(layer.mlp.register_forward_hook(get_activation(f'ffn_layer_{i}')))

# 生成n个token并收集激活值
generated = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    pad_token_id=tokenizer.pad_token_id,
    max_new_tokens=args.n_tokens,
    output_attentions=False,
    output_hidden_states=False,
    return_dict_in_generate=True
)

# 收集每个token位置的激活值
for token_idx in range(args.n_tokens):
    # 获取当前token的输入
    current_input = {
        'input_ids': generated.sequences[:, :inputs['input_ids'].shape[1]+token_idx],
        'attention_mask': torch.ones_like(generated.sequences[:, :inputs['input_ids'].shape[1]+token_idx])
    }

    # 运行模型推理
    outputs = model(**current_input)
    
    # 收集激活值并累加
    current_embedding = model.model.embed_tokens(current_input['input_ids']).detach().cpu().numpy().mean(axis=1)
    # 如果是muP模型，对embedding激活值进行缩放
    if hasattr(model.config, 'mup_input_scale'):
        current_embedding = current_embedding * model.config.mup_input_scale
    current_logits = outputs.logits.detach().cpu().numpy().mean(axis=1)
    
    if activation_sums['word_embedding'] is None:
        activation_sums['word_embedding'] = current_embedding
        activation_sums['output_logits'] = current_logits
        for i in range(len(model.model.layers)):
            activation_sums['attention_outputs'][i] = activation_data[f'attn_layer_{i}'].mean(axis=1)
            activation_sums['ffn_outputs'][i] = activation_data[f'ffn_layer_{i}'].mean(axis=1)
    else:
        activation_sums['word_embedding'] += current_embedding
        activation_sums['output_logits'] += current_logits
        for i in range(len(model.model.layers)):
            activation_sums['attention_outputs'][i] += activation_data[f'attn_layer_{i}'].mean(axis=1)
            activation_sums['ffn_outputs'][i] += activation_data[f'ffn_layer_{i}'].mean(axis=1)

# 计算平均激活值
activations = {
    'word_embedding': activation_sums['word_embedding'] / args.n_tokens,
    'attention_outputs': [v / args.n_tokens for v in activation_sums['attention_outputs']],
    'ffn_outputs': [v / args.n_tokens for v in activation_sums['ffn_outputs']],
    'output_logits': activation_sums['output_logits'] / args.n_tokens
}

# 移除hook
for hook in hooks:
    hook.remove()


# 计算每层激活值的平均绝对值
def compute_activation_stats(activations):
    stats = {}
    for name, values in activations.items():
        if name.endswith('outputs'):
            stats[name] = [np.abs(v).mean() for v in values]
        else:
            stats[name] = np.abs(values).mean()
    return stats


# 可视化激活值变化
stats = compute_activation_stats(activations)
plt.figure(figsize=(12, 6))

# 绘制attention输出
plt.subplot(1, 3, 1)
plt.plot(stats['attention_outputs'], marker='o')
plt.title('Attention Output Activation')
plt.xlabel('Layer')
plt.ylabel('Mean Abs Activation')

# 绘制FFN输出
plt.subplot(1, 3, 2)
plt.plot(stats['ffn_outputs'], marker='o')
plt.title('FFN Output Activation')
plt.xlabel('Layer')
plt.ylabel('Mean Abs Activation')

# 绘制word embedding和logits
plt.subplot(1, 3, 3)
plt.bar(['Word Embedding', 'Output Logits'],
        [stats['word_embedding'], stats['output_logits']])
plt.title('Embedding/Logits Activation')
plt.ylabel('Mean Abs Activation')

plt.tight_layout()
plt.savefig('activation_stats.png')
plt.show()
