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
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
print(wd)

from models.modeling_xmodel2 import XmodelForCausalLM, XmodelConfig
from utils.data_utils import create_dataloaders


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
    # print(ckps)

    config = XmodelConfig()
    config.vocab_size = 129280
    config.max_position_embeddings = 131072
    config.rope_theta = 500000
    config.intermediate_size = 3840
    config.use_mup = True
    config._attn_implementation = "flash_attention_2"
    print(f'config: {config}')

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    # init a new model from scratch
    model = XmodelForCausalLM(config)
    model.to(device)

    torch.set_default_dtype(default_dtype)

    with open(f'val_loss.jsonl', 'r') as fp:
        lines = fp.readlines()
    
    items = [json.loads(line) for line in lines]
    iters_done = [it['iter'] for it in items]
    ckps = [ckp for ckp in ckps if int(ckp.split('.')[-1]) not in iters_done]

    with open(f'val_loss.jsonl', 'w') as fp:
        for ckp in tqdm(ckps):
            ckpt_path = os.path.join(folder, ckp)
            print(f'processing model {ckpt_path}')
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            iter = int(ckp.split('.')[-1])
            loss = estimate_loss(model)
            json_line = json.dumps(dict(iter=iter, loss=loss))
            print(json_line)
            fp.write(json_line + '\n')


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    data_path = args.data_path
    device = f'cuda:{args.device}'
    eval_iters = 100
    micro_batch_size = 5
    vocab_size = 129280
    max_length = 3776

    # run2: 仅包含wikitext2，仿照MiniCPM论文中Figure 12: Loss curve on C4 dataset
    data_config = [
        ("wikitext2", 1.0000)
    ]

    tokenizer_path = f'tokenizers/deepseekv3/'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # print(tokenizer)
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

    eval(args.ckpt_folder)

