import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
print(wd)

from models.modeling_xmodel2 import XmodelForCausalLM
from val_loss.data_utils import create_dataloaders


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='val_loss/val_data/wikitext2')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ckpt_folder", type=str, default='/data2/liuyang/Xmodel-2.5-history/')
    return parser


def get_batch():
    return next(dataloader_iter).to(device)


@torch.no_grad()
def estimate_loss(model):
    losses = []
    for k in range(eval_iters):
        batch = get_batch()
        loss = model(batch, labels=batch).loss
        losses.append(float(loss.detach().cpu().numpy()))
    return np.mean(losses)


def eval(folder):
    print(f'evaluating...')

    ckps = [f for f in os.listdir(folder) if f.startswith('pytorch_model') and f.endswith('000')]
    ckps = sorted(ckps)
    print(ckps)

    with open(f'{args.run_name}_{segment}.jsonl', 'w') as fp:
        for ckp in tqdm(ckps):
            save_folder = os.path.join(folder, ckp)
            model = XmodelForCausalLM.from_pretrained(save_folder, attn_implementation='flash_attention_2',
                                                      torch_dtype=torch.bfloat16, device_map=device)
            model.eval()
            iter = int(ckp[5:])
            loss = estimate_loss(model)
            json_line = json.dumps(dict(iter=iter, loss=loss)) + '\n'
            fp.write(json_line)


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    data_path = args.data_path
    device = f'cuda:{args.device}'
    eval_iters = 100
    micro_batch_size = 4
    vocab_size = 65280
    max_length = 4096

    # run2: 仅包含wikitext2，仿照MiniCPM论文中Figure 12: Loss curve on C4 dataset
    data_config = [
        ("wikitext2", 1.0000)
    ]

    tokenizer_path = f'tokenizers/deepseekv3/'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(tokenizer)
    print('tokenizer.eos_token_id: ' + str(tokenizer.eos_token_id))

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=max_length,
        data_config=data_config,
        train_data_dir=data_path,
        val_data_dir=None,
        vocab_size=vocab_size,
        eos_token_id=tokenizer.eos_token_id
    )
    dataloader_iter = iter(train_dataloader)

    segment = args.segment

    eval(args.ckpt_folder)

