from transformers import AutoTokenizer

# Load your SentencePiece model
tokenizer = AutoTokenizer.from_pretrained(
    "/data2/liuyang/pretrain_xmodel_i_line/tokenizers/deepseekv3"
)

# Save as tokenizer.json
print(tokenizer.bos_token)
print(tokenizer.bos_token_id)
print(tokenizer.eos_token)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token)
print(tokenizer.pad_token_id)
print(tokenizer.unk_token)
print(tokenizer.unk_token_id)

print(tokenizer.encode('<｜begin▁of▁sentence｜>'))
print(tokenizer.encode('<｜end▁of▁sentence｜>'))
print(tokenizer.decode([0]))
print(tokenizer.decode([1]))
