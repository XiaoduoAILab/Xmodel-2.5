import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
print(wd)


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='/home/data/datasets')
    parser.add_argument("--run_name", type=str, default='run2')

    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    run_name = args.run_name

    x, y = [], []
    start_iter = dict(s1=0, s2=270000, decay=530000)

    with open(f'val_loss.jsonl', 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            data = json.loads(line)
            iter, loss = data['iter'], data['loss']

            if iter <= 270000:
                tokens =iter * 1966080 / 1024 ** 3
            else:
                tokens = 270000 * 1966080 / 1024 ** 3 + (iter - 270000) * 3932160 / 1024 ** 3
            x.append(tokens)
            y.append(loss)

    plt.plot(x, y)
    plt.xlabel('Tokens (B)')
    plt.ylabel('Loss')
    plt.ylim(1.6, 3.0)
    plt.grid(linestyle='dotted')
    plt.savefig('val_loss.pdf')
    plt.show()
