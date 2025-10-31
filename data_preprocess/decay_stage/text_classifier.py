import argparse
import json
import os
import sys
import requests
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(self, base_url="http://localhost:8001"):
        """
        初始化文本分类器
        
        Args:
            base_url: Qwen3-4B模型服务的URL
        """
        self.base_url = base_url
        self.classification_prompt = self._build_classification_prompt()
    
    def _build_classification_prompt(self):
        """
        构建分类prompt
        """
        prompt = """请对以下文本进行分类，按照两个维度：

1. 主题分类（三选一）：
   - 代码（code）：包含编程代码、算法实现、技术文档等
   - 数学（math）：包含数学公式、定理证明、数学问题等
   - 知识（knowledge）：包含百科知识、科学原理、历史人文等

2. 质量评分（1-5分）：
   - 5分：内容完整、逻辑清晰、表达准确、信息丰富
   - 4分：内容较完整、逻辑较清晰、表达较准确
   - 3分：内容基本完整、逻辑基本清晰、表达基本准确
   - 2分：内容不完整、逻辑混乱、表达不准确
   - 1分：内容严重缺失、逻辑严重混乱、表达严重不准确

请严格按照以下JSON格式输出结果：
{
    "topic": "code|math|knowledge",
    "quality": 1|2|3|4|5
}

文本内容：
{text}

分类结果："""
        return prompt
    
    def classify_text(self, text, max_retries=3):
        """
        对单个文本进行分类
        
        Args:
            text: 待分类的文本
            max_retries: 最大重试次数
            
        Returns:
            dict: 分类结果，包含topic和quality字段
        """
        if not text.strip():
            return {"topic": "unknown", "quality": 1}
        
        # 使用字符串替换而不是format，避免文本中的花括号引起错误
        prompt = self.classification_prompt.replace("{text}", text[:4000])  # 限制文本长度
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": "qwen3-4b",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 100,
                        "reasoning": False  # 关闭think mode
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    
                    # 尝试解析JSON结果
                    try:
                        # 提取JSON部分
                        if "{" in content and "}" in content:
                            json_start = content.find("{")
                            json_end = content.rfind("}") + 1
                            json_str = content[json_start:json_end]
                            classification = json.loads(json_str)
                            
                            # 验证结果格式
                            if "topic" in classification and "quality" in classification:
                                topic = classification["topic"]
                                quality = classification["quality"]
                                
                                # 验证topic值
                                if topic not in ["code", "math", "knowledge"]:
                                    logger.warning(f"Invalid topic: {topic}, defaulting to 'knowledge'")
                                    classification["topic"] = "knowledge"
                                
                                # 验证quality值
                                if quality not in [1, 2, 3, 4, 5]:
                                    logger.warning(f"Invalid quality: {quality}, defaulting to 3")
                                    classification["quality"] = 3
                                
                                return classification
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from response: {content}")
                    
                    # 如果JSON解析失败，尝试从文本中提取
                    return self._parse_classification_from_text(content)
                
                else:
                    logger.warning(f"HTTP error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                continue
        
        # 所有重试都失败，返回默认值
        logger.error(f"All retries failed for text classification")
        return {"topic": "unknown", "quality": 3}
    
    def _parse_classification_from_text(self, text):
        """
        从文本响应中解析分类结果
        
        Args:
            text: 模型返回的文本
            
        Returns:
            dict: 解析后的分类结果
        """
        result = {"topic": "knowledge", "quality": 3}
        
        # 尝试提取topic
        topic_keywords = {
            "code": ["代码", "code", "编程", "程序"],
            "math": ["数学", "math", "公式", "定理"],
            "knowledge": ["知识", "knowledge", "百科", "科学"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                result["topic"] = topic
                break
        
        # 尝试提取quality
        for i in range(1, 6):
            if str(i) in text:
                result["quality"] = i
                break
        
        return result

def load_jsonl_data(file_path):
    """
    加载JSONL格式的数据
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        list: 包含所有数据的列表
    """
    logger.info(f"Loading data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
    
    logger.info(f"Loaded {len(data)} items from JSONL file")
    return data

def classify_batch(batch_info):
    """
    对一批数据进行分类（用于多进程处理）
    
    Args:
        batch_info: (batch_id, data_batch, port_offset)
        
    Returns:
        list: 分类结果列表
    """
    batch_id, data_batch, port_offset = batch_info
    port = 8001 + port_offset
    
    logger.info(f"Process {batch_id} started, using port {port}")
    classifier = TextClassifier(base_url=f"http://localhost:{port}")
    
    results = []
    for item in tqdm(data_batch, desc=f"Process {batch_id}", leave=False):
        text = item.get("text", "")
        classification = classifier.classify_text(text)
        
        # 合并原始数据和分类结果
        result_item = item.copy()
        result_item.update(classification)
        results.append(result_item)
    
    return results

def classify_data_parallel(data, num_processes=6):
    """
    使用多进程并行分类数据
    
    Args:
        data: 待分类的数据列表
        num_processes: 进程数量
        
    Returns:
        list: 分类后的数据列表
    """
    logger.info(f"Starting parallel classification with {num_processes} processes")
    
    # 将数据分成多个批次
    batch_size = max(1, len(data) // num_processes)
    batches_with_info = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_id = len(batches_with_info)
        port_offset = batch_id % num_processes  # 循环使用端口
        batches_with_info.append((batch_id, batch, port_offset))
    
    logger.info(f"Split into {len(batches_with_info)} batches, each with ~{batch_size} items")
    
    # 使用多进程池处理批次
    with mp.Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(classify_batch, batches_with_info),
            total=len(batches_with_info),
            desc="Overall classification progress"
        ))
    
    # 合并所有批次的结果
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)
    
    return all_results

def save_classified_data(data, output_path):
    """
    保存分类后的数据为JSONL格式
    
    Args:
        data: 分类后的数据列表
        output_path: 输出文件路径
    """
    logger.info(f"Saving classified data to: {output_path}")
    
    # 检查输出目录是否存在，不存在则创建
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Writing classified data"):
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    logger.info(f"Successfully saved {len(data)} classified items to {output_path}")

def analyze_classification_results(data):
    """
    分析分类结果
    
    Args:
        data: 分类后的数据列表
    """
    logger.info("Analyzing classification results...")
    
    topic_counts = {}
    quality_counts = {}
    
    for item in data:
        topic = item.get("topic", "unknown")
        quality = item.get("quality", 3)
        
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    logger.info("Topic distribution:")
    for topic, count in sorted(topic_counts.items()):
        percentage = (count / len(data)) * 100
        logger.info(f"  {topic}: {count} ({percentage:.2f}%)")
    
    logger.info("Quality distribution:")
    for quality in sorted(quality_counts.keys()):
        count = quality_counts[quality]
        percentage = (count / len(data)) * 100
        logger.info(f"  Quality {quality}: {count} ({percentage:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Text classification using Qwen3-4B models')
    
    parser.add_argument('--input_path', type=str, 
                       default='/data2/liuyang/Xmodel-2.5/sft_mixed_v2_deduped_v4.jsonl',
                       help='Path to the input JSONL file')
    
    parser.add_argument('--output_path', type=str,
                       default='sft_mixed_v2_deduped_v4_classified.jsonl',
                       help='Output JSONL file path for classified data')
    
    parser.add_argument('--num_processes', type=int,
                       default=6,
                       help='Number of processes for parallel classification (default: 6)')
    
    parser.add_argument('--sample_size', type=int,
                       default=None,
                       help='Sample size for testing (process only first N items)')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        data = load_jsonl_data(args.input_path)
        
        # 如果指定了样本大小，只处理前N个样本
        if args.sample_size and args.sample_size < len(data):
            logger.info(f"Sampling first {args.sample_size} items for testing")
            data = data[:args.sample_size]
        
        # 并行分类数据
        classified_data = classify_data_parallel(data, args.num_processes)
        
        # 分析分类结果
        analyze_classification_results(classified_data)
        
        # 保存分类后的数据
        save_classified_data(classified_data, args.output_path)
        
        logger.info("Text classification completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during text classification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
