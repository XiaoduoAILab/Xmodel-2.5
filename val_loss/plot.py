import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
print(wd)


if __name__ == "__main__":
    x, y = [], []

    with open(f'val_loss.jsonl', 'r') as fp:
        lines = fp.readlines()
        items = [json.loads(line) for line in lines]
        items = sorted(items, key=lambda x: x['iter'])
        for item in items:
            iter, loss = item['iter'], item['loss']

            if iter <= 270000:
                tokens =iter * 1812480 / 1024 ** 3
            else:
                tokens = 270000 * 1812480 / 1024 ** 3 + (iter - 270000) * 3624960 / 1024 ** 3
            x.append(tokens)
            y.append(loss)

    plt.plot(x, y)
    plt.xlabel('Tokens (B)')
    plt.ylabel('Loss')
    plt.ylim(2.2, 3.5)
    plt.grid(linestyle='dotted')
    plt.savefig('val_loss.pdf')
    plt.show()
