import glob
import os

from utils import sample_json_data

if __name__ == "__main__":
    input_dir = "/data6/datasets/fineweb-edu/data/"
    output_dir = "/data13/liuyang/v12_corpus/"
    json_files = []
    for pattern in ["algebraic-stack-train-*.json",
                    "arxiv-*.json",
                    "books-*.json",
                    "megawika-*.json",
                    "open-web-math-train-*.json",
                    "pes2o-*.json",
                    "reddit-*.json",
                    "stackexchange-*.json",
                    "tulu_*.json",
                    "wiki-*.json"]:
        json_files += glob.glob(os.path.join('/data6/datasets/dolma/target/', pattern))
    output_file = "dolma_wo_cc.txt"
    target_size_gb = 4.0
    sample_json_data(json_files=json_files, output_file=output_file, output_dir=output_dir,
                     target_size_gb=target_size_gb)
