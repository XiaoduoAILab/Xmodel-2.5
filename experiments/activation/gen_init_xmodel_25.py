import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from models.configuration_xmodel2 import XmodelConfig
from models.modeling_xmodel2 import XmodelForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model')
    # general
    parser.add_argument('--model', type=str, default='xl')
    parser.add_argument('--vocab_size', type=int, default=129280)
    parser.add_argument('--output_folder', type=str, default='/data2/liuyang/models/Xmodel2.5/random_init')
    args = parser.parse_args()

    dst_dir = args.output_folder

    # Ensure destination folder exists
    os.makedirs(dst_dir, exist_ok=True)

    config = XmodelConfig.from_name(args.model)
    config.vocab_size = args.vocab_size
    config.torch_dtype = "bfloat16"

    config.initializer_range = 1.0111625715206025
    config.use_mup = True
    config.mup_input_scale = 1.0111625715206025
    config.mup_output_scale = 6.107800284204766
    config.mup_attention_residual_scale = 43.75496659293878
    config.mup_ffn_residual_scale = 0.0013289544687598812

    model = XmodelForCausalLM(config)
    model = model.to(torch.bfloat16)
    model.save_pretrained(args.output_folder, safe_serialization=False, max_shard_size="20GB")
    print(f"Model saved to {args.output_folder}")
    # Copy models/configuration_xmodel2.py and models/modeling_xmodel2.py to the output folder
    files_to_copy = [
        os.path.join('models', 'configuration_xmodel2.py'),
        os.path.join('models', 'modeling_xmodel2.py'),
        os.path.join('tokenizers', 'deepseekv3', 'tokenizer_config.json'),
        os.path.join('tokenizers', 'deepseekv3', 'tokenizer.json'),
    ]

    for file_path in files_to_copy:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
        shutil.copy2(file_path, dst_dir)
        print(f"Copied {file_path} -> {dst_dir}")

    with open(os.path.join(dst_dir, "config.json"), "r") as f:
        data = json.load(f)
        data["_attn_implementation"] = "eager"

    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(data, f, indent=2)
