import re
import json

def clean_generation(text: str) -> str:
    # 去掉 Qwen3 chat template 的标记
    text = text.split("<|im_start|>assistant")[-1]
    text = text.split("<|im_end|>")[0]
    # 去掉首尾空白、空行
    # text = text.strip()
    # 如果前面有多余的换行或注释，也一并去掉
    # text = re.sub(r'^\s*\n', '', text)
    return text

if __name__ == "__main__":
    file_path = "/data1/liuyang/bigcode-evaluation-harness/generations_humaneval.json"
    with open(file_path, "r") as f:
        data = json.load(f) 
    
    for item in data:
        raw_generation = item[0]
        cleaned = clean_generation(raw_generation)
        print("raw_generation:\n", raw_generation)
        print("cleaned:\n", cleaned)
        break
        

    # 假设 raw_generation 是模型输出的原始字符串
    #cleaned = clean_generation(raw_generation)