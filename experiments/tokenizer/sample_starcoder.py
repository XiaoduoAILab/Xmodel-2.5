import glob

from utils import sample_json_data

if __name__ == "__main__":
    input_dir = "/data6/datasets/fineweb-edu/data/"
    output_dir = "/data13/liuyang/v12_corpus/"
    json_files = glob.glob("/data6/datasets/dolma/target/starcoder-*.json")
    output_file = "starcoder.txt"
    target_size_gb = 10.4
    sample_json_data(json_files=json_files, output_file=output_file, output_dir=output_dir,
                     target_size_gb=target_size_gb)
