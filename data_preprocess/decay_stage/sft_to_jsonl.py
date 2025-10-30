import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import multiprocessing as mp
from functools import partial

def load_binary_data(file_path):
    """
    加载二进制数据文件
    """
    print(f"Loading binary data from: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Binary file not found: {file_path}")
    
    # 使用numpy memmap加载二进制数据，dtype=np.uint16
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    
    print(f"Loaded {len(data)} tokens from binary file")
    return data

def split_documents_by_eos(token_ids, eos_token_id=2):
    """
    根据EOS token分割文档
    """
    print("Splitting documents by EOS token...")
    
    # 找到所有EOS token的位置
    eos_positions = np.where(token_ids == eos_token_id)[0]
    
    documents = []
    start_idx = 0
    
    for eos_pos in tqdm(eos_positions, desc="Splitting documents"):
        # 获取从start_idx到eos_pos的token序列（包含EOS token）
        doc_tokens = token_ids[start_idx:eos_pos+1]
        documents.append(doc_tokens)
        start_idx = eos_pos + 1
    
    # 添加最后一个文档（如果没有以EOS结尾）
    if start_idx < len(token_ids):
        doc_tokens = token_ids[start_idx:]
        documents.append(doc_tokens)
    
    print(f"Split into {len(documents)} documents")
    return documents

def decode_single_document(tokenizer_path, doc_tokens):
    """
    解码单个文档（用于多进程处理）
    """
    try:
        # 在每个进程中加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        # 解码为文本，跳过特殊token
        decoded_text = tokenizer.decode(doc_tokens.tolist(), skip_special_tokens=True)
        return decoded_text
    except Exception as e:
        print(f"Error decoding document: {e}")
        return ""

def decode_documents_with_tokenizer_parallel(tokenizer_path, documents, num_processes=None):
    """
    使用多进程并行解码文档
    """
    print("Decoding documents with tokenizer (parallel)...")
    
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(documents))
    
    print(f"Using {num_processes} processes for decoding")
    
    # 准备参数
    decode_func = partial(decode_single_document, tokenizer_path)
    
    # 使用多进程池
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(decode_func, documents),
            total=len(documents),
            desc="Decoding documents"
        ))
    
    return results

def decode_documents_with_tokenizer(tokenizer, documents):
    """
    使用tokenizer解码文档（单进程版本，向后兼容）
    """
    print("Decoding documents with tokenizer (single process)...")
    
    decoded_documents = []
    
    for i, doc_tokens in enumerate(tqdm(documents, desc="Decoding documents")):
        try:
            # 解码为文本，跳过特殊token
            decoded_text = tokenizer.decode(doc_tokens.tolist(), skip_special_tokens=True)
            decoded_documents.append(decoded_text)
        except Exception as e:
            print(f"Error decoding document {i}: {e}")
            decoded_documents.append("")  # 错误时添加空文档
    
    return decoded_documents

def split_into_documents(text, max_doc_length=10000):
    """
    将长文本分割成文档
    """
    print("Splitting text into documents...")
    
    # 简单的分割策略：按段落分割
    paragraphs = text.split('\n\n')
    documents = []
    current_doc = []
    current_length = 0
    
    for para in tqdm(paragraphs, desc="Splitting documents"):
        para_length = len(para)
        
        # 如果当前文档为空，直接添加段落
        if current_length == 0:
            current_doc.append(para)
            current_length = para_length
        # 如果添加这个段落不会超过最大长度，则添加到当前文档
        elif current_length + para_length <= max_doc_length:
            current_doc.append(para)
            current_length += para_length
        # 否则，保存当前文档并开始新文档
        else:
            if current_doc:  # 确保当前文档不为空
                documents.append('\n\n'.join(current_doc))
            current_doc = [para]
            current_length = para_length
    
    # 添加最后一个文档
    if current_doc:
        documents.append('\n\n'.join(current_doc))
    
    print(f"Split into {len(documents)} documents")
    return documents

def save_as_jsonl(documents, output_path):
    """
    将文档保存为JSONL格式
    """
    print(f"Saving documents to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in tqdm(documents, desc="Writing JSONL"):
            json_line = json.dumps({"text": doc}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Successfully saved {len(documents)} documents to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert Xmodel-2 SFT binary file to JSONL format')
    
    parser.add_argument('--data_path', type=str, 
                       default='/data2/wangqun/g_line_data/_SFT_mixed_v2_deduped_v4.bin',
                       help='Path to the SFT binary file')
    
    parser.add_argument('--tokenizer_path', type=str,
                       default='tokenizers/xmodel/v11',
                       help='Path to the tokenizer directory')
    
    parser.add_argument('--output_path', type=str,
                       default='sft_mixed_v2_deduped_v4.jsonl',
                       help='Output JSONL file path')
    
    parser.add_argument('--eos_token_id', type=int,
                       default=2,
                       help='EOS token ID used to separate documents')
    
    parser.add_argument('--num_processes', type=int,
                       default=None,
                       help='Number of processes for parallel decoding (default: CPU count)')
    
    parser.add_argument('--use_parallel', action='store_true',
                       help='Use parallel processing for decoding (recommended for large datasets)')
    
    args = parser.parse_args()
    
    # 检查输出目录是否存在，不存在则创建
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载二进制数据
        print(f"Loading binary data from: {args.data_path}")
        token_ids = load_binary_data(args.data_path)
        
        # 根据EOS token分割文档
        documents_tokens = split_documents_by_eos(token_ids, args.eos_token_id)
        
        # 解码文档
        if args.use_parallel:
            print("Using parallel decoding...")
            documents = decode_documents_with_tokenizer_parallel(
                args.tokenizer_path, 
                documents_tokens, 
                args.num_processes
            )
        else:
            print("Using single process decoding...")
            # 加载tokenizer
            print(f"Loading tokenizer from: {args.tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
            print("Tokenizer loaded successfully")
            documents = decode_documents_with_tokenizer(tokenizer, documents_tokens)
        
        # 过滤空文档
        non_empty_documents = [doc for doc in documents if doc.strip()]
        print(f"Filtered {len(documents) - len(non_empty_documents)} empty documents")
        
        # 保存为JSONL格式
        save_as_jsonl(non_empty_documents, args.output_path)
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
