from utils import sample_parquet_data

if __name__ == "__main__":
    input_dir = "/data11/datasets/Fineweb-Edu-Chinese-V2.1/4_5"
    output_dir = "/data13/liuyang/v12_corpus/"
    output_file = "fineweb_edu_chinese_v2.1.txt"
    ratio = 0.4  # UTF-8 文本文件字符数和文件大小（字节）之比
    target_size_gb = 8.0
    sample_parquet_data(input_dir=input_dir,
                        output_file=output_file,
                        output_dir=output_dir,
                        target_size_gb=target_size_gb,
                        ratio=ratio)
