import json
import os

import pyarrow.parquet as pq
from tqdm import tqdm


def get_file_sizes(parquet_files):
    """Calculate file sizes using os.path.getsize for faster estimation."""
    file_sizes = []
    total_size = 0

    for file_path in parquet_files:
        try:
            size = os.path.getsize(file_path)
            file_sizes.append((file_path, size))
            total_size += size
        except Exception as e:
            print(f"Error getting size for {file_path}: {e}")
            continue

    return total_size, file_sizes


def fetch_samples(file_path, quota, text_field):
    parquet_file = pq.ParquetFile(file_path)

    sampled_texts = []
    sampled = 0

    # 逐批读取数据
    for batch in parquet_file.iter_batches(batch_size=1000):  # 可以调整batch_size
        # 将批数据转换为pandas DataFrame(小批量)
        df_batch = batch.to_pandas()

        # 逐行处理
        for _, row in df_batch.iterrows():
            # 在这里处理每一行数据
            text = str(row[text_field])

            sampled_texts.append(text)
            sampled += len(text) + 1
            if sampled >= quota:
                return sampled_texts, sampled

    return sampled_texts, sampled


def process_parquet_file(file_path, quota, output_file, text_field):
    try:
        sampled_texts, sampled = fetch_samples(file_path, quota, text_field)

        with open(output_file, 'a', encoding='utf-8') as f_out:
            for text in sampled_texts:
                f_out.write(text + '\n')
        return sampled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def sample_parquet_data(input_dir, output_dir, output_file, target_size_gb, text_field="text", ratio=1.0):
    """
    Uniformly sample text data from parquet files with deduplication (parallel version).

    Args:
        input_dir: Directory containing parquet files
        output_file: Output text file path
        target_size_gb: Target size in GB
        text_field: Field name containing text
    """
    # Convert GB to bytes
    target_size = int(target_size_gb * 1024 ** 3)
    output_file = os.path.join(output_dir, output_file)

    # Get all parquet files recursively
    parquet_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))

    # Calculate file sizes and sampling quotas
    total_size, file_sizes = get_file_sizes(parquet_files)
    if total_size == 0:
        raise ValueError("No valid text data found in parquet files")

    # Calculate target samples per file based on size proportion
    file_quotas = []
    for file_path, size in file_sizes:
        quota = int(target_size * (size / total_size) * ratio)
        file_quotas.append((file_path, quota))

    # Initialize output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        pass

    # Setup parallel processing
    results = []
    for fp, q in tqdm(file_quotas):
        written = process_parquet_file(fp, q, output_file, text_field)
        results.append(written)

    total_written = sum(results)
    print(f"Total written: {total_written / 1024 ** 3:.2f} billion characters")


def process_json_file(file_path, quota, output_file, text_field):
    try:
        sampled_texts = []
        sampled = 0
        with open(file_path) as fp:
            while True:
                line = fp.readline()
                item = json.loads(line)
                text = item[text_field]
                sampled_texts.append(text)
                sampled += len(text) + 1
                if sampled >= quota:
                    break

        written = 0
        with open(output_file, 'a', encoding='utf-8') as f_out:
            for text in sampled_texts:
                f_out.write(text + '\n')
                written += len(text) + 1
        return written
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def sample_json_data(json_files, output_dir, output_file, target_size_gb, text_field="text"):
    # Convert GB to bytes
    target_size = int(target_size_gb * 1024 ** 3)
    output_file = os.path.join(output_dir, output_file)

    # Calculate file sizes and sampling quotas
    total_size, file_sizes = get_file_sizes(json_files)
    if total_size == 0:
        raise ValueError("No valid text data found in parquet files")

    # Calculate target samples per file based on size proportion
    ratio = 1.0  # UTF-8 文本文件字符数和文件大小（字节）之比
    file_quotas = []
    for file_path, size in file_sizes:
        quota = int(target_size * (size / total_size) * ratio)
        file_quotas.append((file_path, quota))

    # Initialize output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        pass

    # Setup parallel processing
    results = []
    for fp, q in tqdm(file_quotas):
        written = process_json_file(fp, q, output_file, text_field)
        results.append(written)

    total_written = sum(results)
    print(f"Total written: {total_written / 1024 ** 3:.2f} billion characters")
