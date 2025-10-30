from transformers import AutoTokenizer

# 使用 T5Tokenizer 加载 SentencePiece 模型
tokenizer = AutoTokenizer.from_pretrained(
    "tokenizers/sentencepiece",  # 使用通用 SentencePiece 处理器
    vocab_file="xmodel_65536.model",  # 您的模型文件
    unk_token="<unk>",  # 根据实际设置
    pad_token="<pad>",  # 根据实际设置
    bos_token="<s>",    # 根据实际设置
    eos_token="</s>"    # 根据实际设置
)

# 保存为 tokenizer.json
tokenizer.save_pretrained("tokenizers/xmodel/v12/")
