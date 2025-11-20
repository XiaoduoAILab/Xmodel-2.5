import glob
import os
import random
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class CurriculumDataset(IterableDataset):
    def __init__(self, batch_size, block_size, data_dir, data_config, weights, shuffle, vocab_size, eos_token_id):
        self._batch_size = batch_size
        self._block_size = block_size
        self._data_dir = data_dir
        self._data_config = data_config
        self._weights = weights
        self._shuffle = shuffle
        self._vocab_size = vocab_size
        self._eos_token_id = eos_token_id

    def __iter__(self):
        return CurriculumDatasetIterator(
            batch_size=self._batch_size,
            block_size=self._block_size,
            data_dir=self._data_dir,
            data_config=self._data_config,
            weights=self._weights,
            shuffle=self._shuffle,
            vocab_size=self._vocab_size,
            eos_token_id=self._eos_token_id
        )


class CurriculumDatasetIterator:
    def __init__(self, batch_size, block_size, data_dir, data_config, weights, shuffle, vocab_size, eos_token_id):
        self._batch_size = batch_size
        self._block_size = block_size
        self._data_dir = data_dir
        self._data_config = data_config
        self._weights = weights
        self._shuffle = shuffle
        self._eos_token_id = eos_token_id

        self._rng = random.Random()

        datasets = []
        for i, (prefix, _) in enumerate(data_config):
            weight = weights[i]
            filenames = glob.glob(os.path.join(data_dir, prefix + "*.bin"))
            item = {"prefix": prefix, "filenames": filenames, "weight": weight}
            datasets.append(item)
            # print(item)
        # print('datasets: ' + str(datasets))

        self._datasets = datasets

        self._mmaps = dict()
        self._dtype = np.uint32


    def _close_mmap(self):
        for mmap in self._mmaps.values():
            mmap._mmap.close()

    def __del__(self):
        self._close_mmap()
        del self._mmaps

    def __iter__(self):
        return self

    # liuyangfoam, 2024/11/25, 样本对齐
    def sample_with_alignment(self, ix, mmap):
        found = False
        offset = -1
        for j in range(self._block_size):
            if mmap[ix + j] == self._eos_token_id:
                found = True
                offset = j

        if found and ix + offset + self._block_size < len(mmap):
            new_start = ix + offset + 1
            # sample = torch.from_numpy((mmap[new_start:new_start + self._block_size]).astype(np.int64))
            sample = torch.tensor(mmap[new_start:new_start + self._block_size], dtype=torch.long)
        else:
            # sample = torch.from_numpy((mmap[ix:ix + self._block_size]).astype(np.int64))
            sample = torch.tensor(mmap[ix:ix + self._block_size], dtype=torch.long)
        return sample

    def __next__(self):
        x = torch.empty(size=(self._batch_size, self._block_size), dtype=torch.int64)

        for i in range(self._batch_size):
            dataset = self._rng.choices(self._datasets, weights=self._weights, k=1)[0]
            # print(dataset)
            filename = self._rng.choices(dataset['filenames'], k=1)[0]
            # print(filename)

            if filename not in self._mmaps:
                mmap = np.memmap(filename, mode="r", dtype=self._dtype, offset=0, order="C")
                self._mmaps[filename] = mmap
            else:
                mmap = self._mmaps[filename]

            ix = random.randint(0, len(mmap) - self._block_size - 1)
            # x[i] = torch.from_numpy((mmap[ix:ix + self._block_size]).astype(np.int64))
            x[i] = self.sample_with_alignment(ix, mmap)

        # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
        # x = x.pin_memory().to('cuda', non_blocking=True)
        return x


def create_dataloader(
        batch_size: int,
        block_size: int,
        data_dir: str,
        data_config,
        shuffle: bool = True,
        vocab_size: int = 32000,
        eos_token_id: int = 2
) -> DataLoader:
    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    curriculum_dataset = CurriculumDataset(batch_size=batch_size,
                                           block_size=block_size,
                                           data_dir=data_dir,
                                           data_config=data_config,
                                           weights=weights,
                                           shuffle=shuffle,
                                           vocab_size=vocab_size,
                                           eos_token_id=eos_token_id)
    return DataLoader(curriculum_dataset,
                      batch_size=None,
                      shuffle=False,
                      pin_memory=True,
                      num_workers=2,
                      prefetch_factor=2
                      )


def create_dataloaders(
        batch_size: int,
        block_size: int,
        data_config,
        train_data_dir: str = "/data2/wangqun/f_line_data_v11_65280/",
        val_data_dir: Optional[str] = None,
        vocab_size: int = 32000,
        eos_token_id: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    data_config descript the data source and data sample ratio e.g. ("wikipedia", 4.5), ("arxiv", 2.5),
     """
    # Increase by one because we need the next word as well
    # effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=block_size,
        data_dir=train_data_dir,
        data_config=data_config,
        shuffle=True,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=block_size,
            data_dir=val_data_dir,
            data_config=data_config,
            shuffle=False,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    data_config = [
        ("SFT_mixed_dedup_v6", 1.0000),
    ]

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=3,
        block_size=32768,
        data_config=data_config,
        train_data_dir='/home/data/datasets',
        val_data_dir=None,
        vocab_size=65280
    )

    # training loop
    dataloader_iter = iter(train_dataloader)
    batch = next(dataloader_iter)
    print(batch.shape)
    print(batch)
