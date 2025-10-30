# SFT Binary to JSONL Converter

这个脚本用于将 Xmodel-2 的 SFT_mixed 二进制文件转换为 JSONL 格式。

## 功能

- 读取 numpy memmap 格式的二进制文件 (.bin)
- 使用 EOS token (默认ID=2) 分割文档
- 使用指定的 tokenizer 将 token IDs 解码为文本
- 将文档保存为 JSONL 格式，每行包含一个 `text` 字段

## 使用方法

### 基本用法

```bash
python sft_to_jsonl.py
```

使用默认参数：
- 数据路径：`/data2/wangqun/g_line_data/_SFT_mixed_v2_deduped_v4.bin`
- Tokenizer路径：`tokenizers/xmodel/v11`
- 输出路径：`sft_mixed_v2_deduped_v4.jsonl`
- EOS token ID：`2`

### 自定义参数

```bash
python sft_to_jsonl.py \
    --data_path /path/to/your/data.bin \
    --tokenizer_path /path/to/tokenizer \
    --output_path /path/to/output.jsonl \
    --eos_token_id 2
```

### 参数说明

- `--data_path`: 二进制数据文件路径（包含 .bin 扩展名）
- `--tokenizer_path`: tokenizer 目录路径
- `--output_path`: 输出的 JSONL 文件路径
- `--eos_token_id`: 用于分割文档的 EOS token ID（默认：2）

## 文件格式

二进制数据文件格式：
- 使用 `numpy.memmap` 存储，`dtype=np.uint16`
- 文档之间使用 EOS token (ID=2) 分隔
- 每个文档以 EOS token 结尾

## 依赖

- transformers
- torch
- numpy
- tqdm
