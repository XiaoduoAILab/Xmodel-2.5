import argparse
import json
import os.path

from datasets import load_dataset
from tqdm import tqdm


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str,
                        default='/data12/datasets/institutional-books-1.0/data/train-00000-of-09831.parquet ')
    parser.add_argument("--output_dir", type=str, default='/data13/datasets/pretrain/institutional-books_json/')
    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    data_path = args.data_path
    end = data_path.rindex('/')
    start = data_path[:end].rindex('/') + 1
    prefix = data_path[start: end]

    basename = os.path.basename(data_path)
    name, ext = os.path.splitext(basename)
    basename = basename.replace(ext, '.json')
    basename = f'{prefix}_{basename}'
    file_path = os.path.join(args.output_dir, basename)

    if not os.path.isfile(file_path):

        dataset = load_dataset(
            "parquet",
            data_files=data_path,
            split="train",
            # streaming=True,
        )

        lines = []
        for example in tqdm(dataset):
            if example['text_by_page_gen']:
                lines.append(json.dumps(dict(text='\n'.join(example['text_by_page_gen']))) + '\n')

        with open(file_path, 'w', encoding='utf-8') as fp:
            fp.writelines(lines)
