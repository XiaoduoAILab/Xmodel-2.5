from utils import sample_parquet_data

if __name__ == "__main__":
    input_dir = "/data11/datasets/finemath/finemath-4plus/"
    output_dir = "/data13/liuyang/v12_corpus/"
    output_file = "finemath.txt"
    target_size_gb = 0.4
    ratio = 1.0
    sample_parquet_data(input_dir=input_dir,
                        output_file=output_file,
                        output_dir=output_dir,
                        target_size_gb=target_size_gb,
                        ratio=ratio)
