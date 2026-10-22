#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION.  Convert Huggingface checkpoint to Megatron-LM mcore format
#
# Usage:
#   python tools/checkpoint/hf_to_megatron_mup.py \
#       --hf-checkpoint-path /path/to/step_532000/pytorch_model.bin \
#       --save-dir /path/to/megatron_checkpoint \
#       --iteration 532000
#
# The output directory will contain:
#   <save-dir>/iter_0532000/mp_rank_00/model_optim_rng.pt
#   <save-dir>/latest_checkpointed_iteration.txt
#
# The resulting checkpoint can be loaded with:
#   --load <save-dir> --no-load-optim     (training script default, loads weights+iteration)
#   --load <save-dir> --finetune          (loads weights only, resets iteration to 0)
#   --load <save-dir> --no-load-optim --no-load-rng  (loads weights+iteration, no RNG)

import argparse
import copy
import os
import sys
import pickle

import torch


def build_te_model_for_extra_state(num_layers=2):
    """Build a small GPTModel with Transformer Engine on GPU to extract valid _extra_state templates.

    Returns a dict mapping _extra_state key suffixes to their template values.
    """
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    import torch.nn.functional as F

    # Initialize minimal parallel state for single GPU
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(0)
        try:
            torch.distributed.init_process_group(
                backend='nccl', init_method='tcp://localhost:0',
                world_size=1, rank=0,
            )
        except Exception:
            try:
                torch.distributed.init_process_group(
                    backend='gloo', init_method='env://',
                    world_size=1, rank=0,
                )
            except Exception:
                pass  # May already be initialized

    if not mpu.is_initialized():
        mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=1536,
        num_attention_heads=24,
        num_query_groups=8,
        ffn_hidden_size=3840,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        perform_initialization=False,
        normalization='RMSNorm',
        layernorm_epsilon=1e-5,
        tp_comm_overlap=False,
        fp8='hybrid',
        fp8_recipe='delayed',
        fp8_amax_history_len=128,
        fp8_amax_compute_algo='max',
        fp8_wgrad=True,
        fp8_dot_product_attention=False,
        fp8_multi_head_attention=False,
        add_bias_linear=False,
        add_qkv_bias=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        init_method_std=0.1,
        layernorm_zero_centered_gamma=False,
        attention_backend=AttnBackend.fused,
        activation_func=F.silu,
        gated_linear_unit=True,
    )

    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None, qk_layernorm=False, qk_l2_norm=False,
    )

    model = GPTModel(
        config=config, transformer_layer_spec=spec,
        vocab_size=129408, max_sequence_length=3776,
        pre_process=True, post_process=True,
        share_embeddings_and_output_weights=True,
    )
    model = model.cuda().to(torch.bfloat16)

    sd = model.state_dict_for_save_checkpoint()

    extra_state_templates = {}
    for key, val in sd.items():
        if '_extra_state' in key and 'layers.0.' in key:
            suffix = key.replace('decoder.layers.0.', 'decoder.layers.{i}.')
            extra_state_templates[suffix] = val
    for key in ['output_layer._extra_state', 'embedding.word_embeddings._extra_state',
                 'decoder.final_layernorm._extra_state']:
        if key in sd:
            extra_state_templates[key] = sd[key]

    print(f"Extracted {len(extra_state_templates)} _extra_state templates:")
    for k, v in extra_state_templates.items():
        vtype = type(v).__name__
        vshape = v.shape if hasattr(v, 'shape') else 'N/A'
        vdtype = v.dtype if hasattr(v, 'dtype') else 'N/A'
        print(f"  {k}: type={vtype}, shape={vshape}, dtype={vdtype}")

    # Clean up GPU memory
    del model, sd
    torch.cuda.empty_cache()

    return extra_state_templates


def convert_hf_to_megatron(hf_checkpoint_path, save_dir, iteration=532000, num_layers=48):
    """Convert HF checkpoint to Megatron mcore format."""

    hidden_size = 1536
    num_attention_heads = 24
    query_groups = 8
    head_dim = hidden_size // num_attention_heads  # 64
    value_num_per_group = num_attention_heads // query_groups  # 3
    ffn_hidden_size = 3840

    # 1. Load HF checkpoint
    print(f"Loading HF checkpoint from {hf_checkpoint_path} ...")
    hf_sd = torch.load(hf_checkpoint_path, map_location='cpu', weights_only=False)

    if 'model.embed_tokens.weight' not in hf_sd:
        if 'embed_tokens.weight' in hf_sd:
            hf_sd = {f'model.{k}' if not k.startswith('model.') else k: v
                     for k, v in hf_sd.items()}
        else:
            print("Warning: unexpected key format. First 10 keys:")
            for k in list(hf_sd.keys())[:10]:
                print(f"  {k}")
            raise ValueError("Cannot find 'model.embed_tokens.weight' in HF checkpoint")

    print(f"HF checkpoint loaded. Keys: {len(hf_sd)}")
    print(f"  embed_tokens shape: {hf_sd['model.embed_tokens.weight'].shape}")
    print(f"  dtype: {hf_sd['model.embed_tokens.weight'].dtype}")

    # 2. Build small TE model to get _extra_state templates
    print("\nBuilding TE model on GPU to extract _extra_state templates ...")
    extra_state_templates = build_te_model_for_extra_state(num_layers=2)

    # 3. Convert weights
    print(f"\nConverting {num_layers} layers ...")
    mcore_sd = {}

    # Embedding — pad to data's padded vocab size (129408 = 128*1011)
    # HF model has 129280 (deepseekv3 base), data was preprocessed with 129281-token
    # tokenizer (EOD=129280), padded to 129408. Model tokenizer has 128815 tokens
    # (padded to 128896), but data token IDs go up to 129280 → must pad to 129408.
    MEGATRON_VOCAB_SIZE = 129408  # match data's padded_vocab_size
    embed_weight = hf_sd['model.embed_tokens.weight'].to(torch.bfloat16)
    if embed_weight.shape[0] < MEGATRON_VOCAB_SIZE:
        pad_size = MEGATRON_VOCAB_SIZE - embed_weight.shape[0]
        padding = torch.zeros((pad_size, embed_weight.shape[1]), dtype=embed_weight.dtype)
        embed_weight = torch.cat([embed_weight, padding], dim=0).contiguous()
        print(f"  Padding embedding from {embed_weight.shape[0]-pad_size} to {MEGATRON_VOCAB_SIZE} (data compat)")
    elif embed_weight.shape[0] > MEGATRON_VOCAB_SIZE:
        print(f"  Truncating embedding from {embed_weight.shape[0]} to {MEGATRON_VOCAB_SIZE}")
        embed_weight = embed_weight[:MEGATRON_VOCAB_SIZE].contiguous()
    mcore_sd['embedding.word_embeddings.weight'] = embed_weight
    if 'embedding.word_embeddings._extra_state' in extra_state_templates:
        mcore_sd['embedding.word_embeddings._extra_state'] = \
            extra_state_templates['embedding.word_embeddings._extra_state']

    # Layers
    for i in range(num_layers):
        if i % 8 == 0:
            print(f"  Converting layer {i}/{num_layers} ...")

        prefix_hf = f'model.layers.{i}.'
        prefix_mc = f'decoder.layers.{i}.'

        # --- QKV: interleave Q, K, V by GQA groups ---
        q_weight = hf_sd[f'{prefix_hf}self_attn.q_proj.weight'].to(torch.bfloat16)
        k_weight = hf_sd[f'{prefix_hf}self_attn.k_proj.weight'].to(torch.bfloat16)
        v_weight = hf_sd[f'{prefix_hf}self_attn.v_proj.weight'].to(torch.bfloat16)

        q_reshaped = q_weight.view(query_groups, value_num_per_group, head_dim, hidden_size)
        k_reshaped = k_weight.view(query_groups, 1, head_dim, hidden_size)
        v_reshaped = v_weight.view(query_groups, 1, head_dim, hidden_size)

        qkv = torch.cat([q_reshaped, k_reshaped, v_reshaped], dim=1)
        mcore_sd[f'{prefix_mc}self_attention.linear_qkv.weight'] = \
            qkv.reshape(-1, hidden_size).contiguous()

        mcore_sd[f'{prefix_mc}self_attention.linear_qkv.layer_norm_weight'] = \
            hf_sd[f'{prefix_hf}input_layernorm.weight'].to(torch.bfloat16).contiguous()

        mcore_sd[f'{prefix_mc}self_attention.linear_proj.weight'] = \
            hf_sd[f'{prefix_hf}self_attn.o_proj.weight'].to(torch.bfloat16).contiguous()

        # --- MLP: concat gate + up ---
        gate_weight = hf_sd[f'{prefix_hf}mlp.gate_proj.weight'].to(torch.bfloat16)
        up_weight = hf_sd[f'{prefix_hf}mlp.up_proj.weight'].to(torch.bfloat16)
        mcore_sd[f'{prefix_mc}mlp.linear_fc1.weight'] = \
            torch.cat([gate_weight, up_weight], dim=0).contiguous()

        mcore_sd[f'{prefix_mc}mlp.linear_fc1.layer_norm_weight'] = \
            hf_sd[f'{prefix_hf}post_attention_layernorm.weight'].to(torch.bfloat16).contiguous()

        mcore_sd[f'{prefix_mc}mlp.linear_fc2.weight'] = \
            hf_sd[f'{prefix_hf}mlp.down_proj.weight'].to(torch.bfloat16).contiguous()

        # --- _extra_state from templates ---
        for tmpl_key, tmpl_val in extra_state_templates.items():
            if 'layers.{i}.' in tmpl_key:
                actual_key = tmpl_key.replace('{i}', str(i))
                if isinstance(tmpl_val, torch.Tensor):
                    mcore_sd[actual_key] = tmpl_val.clone()
                else:
                    mcore_sd[actual_key] = copy.deepcopy(tmpl_val)

    # Final layernorm (TransformerBlock.final_layernorm)
    mcore_sd['decoder.final_layernorm.weight'] = hf_sd['model.norm.weight'].to(torch.bfloat16).contiguous()
    if 'decoder.final_layernorm._extra_state' in extra_state_templates:
        tmpl = extra_state_templates['decoder.final_layernorm._extra_state']
        mcore_sd['decoder.final_layernorm._extra_state'] = tmpl.clone() if isinstance(tmpl, torch.Tensor) else tmpl

    # Output layer: with share_embeddings_and_output_weights=True, Megatron does NOT
    # expect output_layer.weight in the state dict (it shares embedding weight).
    # Only include _extra_state if present (it was NoneType in our TE extraction).
    if 'output_layer._extra_state' in extra_state_templates:
        tmpl = extra_state_templates['output_layer._extra_state']
        if tmpl is not None:
            mcore_sd['output_layer._extra_state'] = tmpl.clone() if isinstance(tmpl, torch.Tensor) else tmpl

    print(f"\nMegatron state dict assembled. Keys: {len(mcore_sd)}")

    # 4. Create checkpoint directory structure
    iter_str = f'iter_{iteration:07d}'
    mp_dir = os.path.join(save_dir, iter_str, 'mp_rank_00')
    os.makedirs(mp_dir, exist_ok=True)

    # 5. Build checkpoint state dict
    checkpoint_version = 3.0

    from argparse import Namespace
    checkpoint_args = Namespace(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        add_position_embedding=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        vocab_size=129408,  # match data's padded vocab
        ffn_hidden_size=ffn_hidden_size,
        seq_length=3776,
        max_position_embeddings=131072,
        rotary_base=500000,
        init_method_std=0.1,
        norm_epsilon=1e-5,
        add_qkv_bias=False,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_mup=True,
        mup_base_width=256,
        mup_input_scale=12.0,
        mup_output_scale=1.0,
        mup_attention_residual_scale=1.4,
        mup_ffn_residual_scale=1.4,
        untie_embeddings_and_output_weights=False,
        consumed_train_samples=0,
        skipped_train_samples=0,
        consumed_valid_samples=0,
    )

    checkpoint_state_dict = {
        'model': mcore_sd,  # single dict for single PP rank (not a list)
        'iteration': iteration,
        'args': checkpoint_args,
        'checkpoint_version': checkpoint_version,
        'num_floating_point_operations_so_far': 0,
    }

    # 6. Save checkpoint
    ckpt_path = os.path.join(mp_dir, 'model_optim_rng.pt')
    print(f"\nSaving checkpoint to {ckpt_path} ...")
    torch.save(checkpoint_state_dict, ckpt_path)
    print(f"  Checkpoint size: {os.path.getsize(ckpt_path) / 1e9:.2f} GB")

    # 7. Save tracker file
    tracker_path = os.path.join(save_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_path, 'w') as f:
        f.write(str(iteration))
    print(f"  Tracker file: {tracker_path} -> {iteration}")

    # 8. Verify by loading back
    print("\nVerifying checkpoint by loading back ...")
    verify_sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"  Loaded keys: {list(verify_sd.keys())}")
    print(f"  iteration: {verify_sd['iteration']}")
    model_sd = verify_sd['model']
    print(f"  model type: {type(model_sd).__name__}")
    print(f"  model state dict keys: {len(model_sd)}")

    check_keys = [
        'embedding.word_embeddings.weight',
        'decoder.layers.0.self_attention.linear_qkv.weight',
        'decoder.layers.0.self_attention.linear_qkv.layer_norm_weight',
        'decoder.layers.0.mlp.linear_fc1.weight',
        'decoder.final_layernorm.weight',
        'output_layer.weight',
    ]
    for k in check_keys:
        if k in model_sd:
            v = model_sd[k]
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: MISSING!")

    es_keys = [k for k in model_sd if '_extra_state' in k]
    print(f"  _extra_state keys: {len(es_keys)}")
    for k in es_keys[:3]:
        v = model_sd[k]
        vtype = type(v).__name__
        vshape = v.shape if hasattr(v, 'shape') else 'N/A'
        print(f"    {k}: type={vtype}, shape={vshape}")

    print("\n✅ Conversion complete!")
    print(f"  Checkpoint dir: {save_dir}")
    print(f"  Load with: --load {save_dir} --finetune")
    print(f"  (or: --load {save_dir} --no-load-optim --no-load-rng)")

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HF checkpoint to Megatron mcore format for Xmodel-2.5"
    )
    parser.add_argument(
        "--hf-checkpoint-path", type=str, required=True,
        help="Path to HF pytorch_model.bin or model.safetensors",
    )
    parser.add_argument(
        "--save-dir", type=str, required=True,
        help="Directory to save Megatron checkpoint",
    )
    parser.add_argument(
        "--iteration", type=int, default=532000,
        help="Iteration number for the checkpoint (default: 532000)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=48,
        help="Number of layers in the model (default: 48)",
    )

    args = parser.parse_args()

    convert_hf_to_megatron(
        hf_checkpoint_path=args.hf_checkpoint_path,
        save_dir=args.save_dir,
        iteration=args.iteration,
        num_layers=args.num_layers,
    )

