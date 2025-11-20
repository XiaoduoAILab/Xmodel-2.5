# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import argparse
import os

import numpy as np
import transformers
from datasets import concatenate_datasets, Dataset


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_path", type=str, default='tokenizers/deepseekv3', help="tokenizer path")
    parser.add_argument("--vocab_size", type=int, default=65280, help="vocab size")
    return parser


if __name__ == "__main__":
    num_proc = 8
    num_proc_load_dataset = num_proc

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        skip_special_tokens=True,
        add_bos_token=False,
        add_eos_token=False,
        clean_up_tokenization_spaces=True,
        use_fast=False,
        trust_remote_code=True)
    arrow_files = ["val_loss/val_data/wikitext2/wikitext-test.arrow"]

    dataset = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])["text"]

    encodings = tokenizer("\n\n".join(dataset), return_tensors="pt")
    arr_len = encodings.input_ids.size(1)

    filename = os.path.join(os.path.dirname(__file__), f'wikitext2.bin')
    dtype = np.uint32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    arr[:] = encodings.input_ids[:]
    arr.flush()
