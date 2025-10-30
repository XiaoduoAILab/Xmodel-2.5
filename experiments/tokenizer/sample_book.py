import argparse
import glob

from utils import sample_json_data


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="/data6/datasets/fineweb-edu/data/")
    parser.add_argument("--output_dir", type=str, default="/data13/liuyang/v12_corpus/")
    parser.add_argument("--output_file", type=str, default="book.txt")
    parser.add_argument("--json_files", type=str, default="/data6/datasets/dolma/target/books-*.json")
    parser.add_argument("--target_size_gb", type=float, default=0.4)
    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    json_files = glob.glob(args.json_files)
    sample_json_data(json_files=json_files, output_dir=args.output_dir, output_file=args.output_file,
                     target_size_gb=args.target_size_gb)
