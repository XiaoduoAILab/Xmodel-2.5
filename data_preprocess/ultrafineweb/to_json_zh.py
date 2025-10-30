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
                        default='/data12/datasets/Ultra-FineWeb/data/ultrafineweb_zh/ultrafineweb-zh-part-001-of-256.parquet')
    parser.add_argument("--output_dir", type=str, default='/data13/datasets/pretrain/ultrafineweb-zh_json/')
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
    basename = basename.replace('-of-256', '')
    basename = f'{prefix}_{basename}'
    file_path = os.path.join(args.output_dir, basename)
    file_path = file_path.replace('ultrafineweb_zh_ultrafineweb-zh-', '')

    if not os.path.isfile(file_path):

        dataset = load_dataset(
            "parquet",
            data_files=data_path,
            split="train",
            # streaming=True,
        )

        lines = []
        for example in tqdm(dataset):
            lines.append(json.dumps(example) + '\n')

        with open(file_path, 'w', encoding='utf-8') as fp:
            fp.writelines(lines)
