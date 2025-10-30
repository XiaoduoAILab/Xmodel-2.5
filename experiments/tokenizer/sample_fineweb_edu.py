from utils import sample_parquet_data

if __name__ == "__main__":
    input_dir = "/data6/datasets/fineweb-edu/data/"
    output_dir = "/data13/liuyang/v12_corpus/"
    output_file = "fineweb_edu.txt"
    target_size_gb = 16.4
    sample_parquet_data(input_dir=input_dir,
                        output_dir=output_dir,
                        output_file=output_file,
                        target_size_gb=target_size_gb)
