import argparse
import json
import os
import sys
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)

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
    parser.add_argument("--ckpt_folder", type=str, default='/data2/liuyang/Xmodel-2.5-history/')
    parser.add_argument("--gpu-pool", type=str, help="Comma-separated list of GPU IDs to use as pool, e.g. '0,1,2,3'")
    return parser


def get_batch(dataloader_iter, device):
    return next(dataloader_iter).to(device)


@torch.no_grad()
def estimate_loss(model, dataloader_iter, train_dataloader, device, eval_iters):
    losses = []
    for k in range(eval_iters):
        try:
            batch = get_batch(dataloader_iter, device)
            loss = model(batch, labels=batch).loss
            losses.append(float(loss.detach().cpu().numpy()))
        except StopIteration:
            # Reset dataloader if we run out of data
            dataloader_iter = iter(train_dataloader)
            batch = get_batch(dataloader_iter, device)
            loss = model(batch, labels=batch).loss
            losses.append(float(loss.detach().cpu().numpy()))
    return np.mean(losses)


def eval_checkpoint(args):
    """Evaluate a single checkpoint on a specific GPU"""
    ckpt_path, gpu_id, data_path, eval_iters, micro_batch_size, vocab_size, max_length = args
    
    device = f'cuda:{gpu_id}'
    print(f"Processing {ckpt_path} on GPU {gpu_id}")
    
    try:
        # Initialize tokenizer and dataloader for this process
        tokenizer_path = 'tokenizers/deepseekv3/'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        data_config = [("wikitext2", 1.0000)]
        
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
        
        # Initialize model
        config = XmodelConfig()
        config.vocab_size = 129280
        config.max_position_embeddings = 131072
        config.rope_theta = 500000
        config.intermediate_size = 3840
        config.use_mup = True
        config._attn_implementation = "flash_attention_2"
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        
        model = XmodelForCausalLM(config)
        model.to(device)
        
        torch.set_default_dtype(default_dtype)
        
        # Load checkpoint and evaluate
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        iter_num = int(os.path.basename(ckpt_path).split('.')[-1])
        loss = estimate_loss(model, dataloader_iter, train_dataloader, device, eval_iters)
        
        result = {
            'iter': iter_num,
            'loss': loss,
            'gpu': gpu_id
        }
        
        # Append result to shared file
        with open('val_loss.jsonl', 'a') as fp:
            json_line = json.dumps(result)
            fp.write(json_line + '\n')
            fp.flush()
        
        print(f"GPU {gpu_id}: iter={iter_num}, loss={loss:.4f}")
        
        # Clean up to prevent memory leaks
        del model
        del state_dict
        del train_dataloader
        del val_dataloader
        del dataloader_iter
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"Error evaluating checkpoint {ckpt_path} on GPU {gpu_id}: {e}")
        # Clean up even if there's an error
        if 'model' in locals():
            del model
        if 'state_dict' in locals():
            del state_dict
        torch.cuda.empty_cache()
        raise e


def get_pending_checkpoints(ckpt_folder):
    """Get list of checkpoints that haven't been evaluated yet"""
    ckps = [f for f in os.listdir(ckpt_folder) if f.startswith('pytorch_model') and f.endswith('000')]
    ckps = sorted(ckps)
    
    # Read existing results to filter out already evaluated checkpoints
    evaluated_iters = set()
    if os.path.exists('val_loss.jsonl'):
        with open('val_loss.jsonl', 'r') as fp:
            for line in fp:
                try:
                    data = json.loads(line.strip())
                    evaluated_iters.add(data['iter'])
                except:
                    continue
    
    pending_checkpoints = []
    for ckp in ckps:
        iter_num = int(ckp.split('.')[-1])
        if iter_num not in evaluated_iters:
            pending_checkpoints.append(os.path.join(ckpt_folder, ckp))
    
    print(f"Found {len(pending_checkpoints)} pending checkpoints out of {len(ckps)} total")
    return pending_checkpoints


def main():
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    data_path = args.data_path
    eval_iters = 100
    micro_batch_size = 5
    vocab_size = 129280
    max_length = 3776

    # Determine GPU pool
    if args.gpu_pool:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_pool.split(',')]
    else:
        gpu_ids = [0]  # Default to GPU 0 if no pool specified
    
    print(f"Using GPU pool: {gpu_ids}")
    num_processes = len(gpu_ids)
    
    # Get pending checkpoints
    pending_checkpoints = get_pending_checkpoints(args.ckpt_folder)
    
    if not pending_checkpoints:
        print("No pending checkpoints to evaluate")
        return
    
    # Prepare arguments for each checkpoint
    task_args = []
    for i, ckpt_path in enumerate(pending_checkpoints):
        gpu_id = gpu_ids[i % num_processes]  # Round-robin assignment
        task_args.append((
            ckpt_path, 
            gpu_id, 
            data_path, 
            eval_iters, 
            micro_batch_size, 
            vocab_size, 
            max_length
        ))
    
    # Create val_loss.jsonl if it doesn't exist
    if not os.path.exists('val_loss.jsonl'):
        with open('val_loss.jsonl', 'w') as fp:
            pass
    
    # Execute in parallel using ProcessPoolExecutor
    print(f"Starting evaluation with {num_processes} processes...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(eval_checkpoint, arg) for arg in task_args]
        
        # Wait for all tasks to complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating checkpoints"):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error in evaluation: {e}")


if __name__ == "__main__":
    main()
