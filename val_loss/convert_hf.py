import argparse
import os
import shutil
import sys
import json
from tqdm import tqdm
import torch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.configuration_xmodel2 import XmodelConfig
from models.modeling_xmodel2 import XmodelForCausalLM


def convert_checkpoint(checkpoint_path, save_dir):
    """Convert a Megatron checkpoint to Huggingface format."""

    # Handle save directory
    if os.path.exists(save_dir):
        file_path = os.path.join(save_dir, "pytorch_model.bin")
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 2700260395:
            print('Directory exists, no need to convert:', save_dir)
            return
    os.makedirs(save_dir, exist_ok=True)

    # Load and merge Megatron-LM checkpoints
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']
    mcore_args = ckpt['args']
    # print(f'mcore_args: {mcore_args}')

    # Initialize Huggingface model config
    config = XmodelConfig()
    config.vocab_size = mcore_args.vocab_size
    config.hidden_size = mcore_args.hidden_size
    config.intermediate_size = mcore_args.ffn_hidden_size
    config.num_hidden_layers = mcore_args.num_layers
    config.num_attention_heads = mcore_args.num_attention_heads
    config.num_key_value_heads = mcore_args.num_query_groups
    config.max_position_embeddings = mcore_args.max_position_embeddings
    config.initializer_range = mcore_args.init_method_std
    config.rms_norm_eps = mcore_args.norm_epsilon
    config.tie_word_embeddings = not mcore_args.untie_embeddings_and_output_weights
    config.rope_theta = mcore_args.rotary_base
    config.attention_bias = mcore_args.add_qkv_bias
    config.attention_dropout = mcore_args.attention_dropout
    config.mlp_bias = mcore_args.add_bias_linear
    config.use_mup = mcore_args.use_mup
    config.mup_input_scale = mcore_args.mup_input_scale
    config.mup_output_scale = mcore_args.mup_output_scale
    config.mup_attention_residual_scale = mcore_args.mup_attention_residual_scale
    config.mup_ffn_residual_scale = mcore_args.mup_ffn_residual_scale
    config.mup_base_width = mcore_args.mup_base_width
    config.mup_width_multiplier = mcore_args.hidden_size / mcore_args.mup_base_width

    # if mcore_args.use_flash_attn:
    config._attn_implementation = "flash_attention_2"

    config.torch_dtype = torch.bfloat16

    # print(f'config: {config}')

    hf_model = XmodelForCausalLM(config)
    # print(f'hf_model: {hf_model}')

    checkpoint_embed = state_dict['embedding.word_embeddings.weight']
    checkpoint_embed_size = checkpoint_embed.shape[0]
    model_embed_size = hf_model.model.embed_tokens.weight.shape[0]

    if checkpoint_embed_size < model_embed_size:
        # Pad checkpoint embeddings with zeros to match model size
        pad_size = model_embed_size - checkpoint_embed_size
        padding = torch.zeros((pad_size, checkpoint_embed.shape[1]), dtype=checkpoint_embed.dtype)
        state_dict['embedding.word_embeddings.weight'] = torch.cat([checkpoint_embed, padding], dim=0)

    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    query_group = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    value_num_per_group = num_attention_heads // query_group
    ffn_hidden_size = config.intermediate_size

    with torch.no_grad():

        hf_model.model.embed_tokens.weight.copy_(state_dict['embedding.word_embeddings.weight'])

        for i in range(num_layers):
            hglayer = hf_model.model.layers[i]
            hglayer.input_layernorm.weight.copy_(
                state_dict[f'decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight'])
            qkv_weight = state_dict[f'decoder.layers.{i}.self_attention.linear_qkv.weight'].view(query_group, -1,
                                                                                                 head_dim, hidden_size)
            q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size_or_sections=[value_num_per_group, 1, 1],
                                                       dim=1)

            hglayer.self_attn.q_proj.weight.copy_(q_weight.reshape(-1, hidden_size))
            hglayer.self_attn.k_proj.weight.copy_(k_weight.reshape(-1, hidden_size))
            hglayer.self_attn.v_proj.weight.copy_(v_weight.reshape(-1, hidden_size))

            hglayer.self_attn.o_proj.weight.copy_(state_dict[f'decoder.layers.{i}.self_attention.linear_proj.weight'])

            gate_weight, fc1_weight = torch.split(state_dict[f'decoder.layers.{i}.mlp.linear_fc1.weight'],
                                                  split_size_or_sections=ffn_hidden_size)
            hglayer.mlp.gate_proj.weight.copy_(gate_weight)
            hglayer.mlp.up_proj.weight.copy_(fc1_weight)
            hglayer.mlp.down_proj.weight.copy_(state_dict[f'decoder.layers.{i}.mlp.linear_fc2.weight'])
            hglayer.post_attention_layernorm.weight.copy_(
                state_dict[f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'])

        hf_model.model.norm.weight.copy_(state_dict[f'decoder.final_layernorm.weight'])
        # hf_model.lm_head.weight.copy_(state_dict[f'decoder.output_layer.weight'])

    # Save Huggingface model
    hf_model = hf_model.to(torch.bfloat16)
    hf_model.save_pretrained(save_dir, safe_serialization=False, max_shard_size="20GB")
    # print(f"Converted checkpoint saved to {save_dir}")

    # 在 megatron_to_hf_mup.py 的 convert_checkpoint() 末尾添加
    # print("Megatron embed mean:", state_dict['embedding.word_embeddings.weight'].mean().item())
    # print("HF embed mean:", hf_model.model.embed_tokens.weight.mean().item())
    # print("Megatron lm_head mean:", state_dict['decoder.output_layer.weight'].mean().item())
    # print("HF lm_head mean:", hf_model.lm_head.weight.mean().item())


def copy_huggingface_tokenizer(dst_path):
    os.system("cp -rf tokenizers/deepseekv3/tokenizer_config.json " + dst_path)
    os.system("cp -rf tokenizers/deepseekv3/tokenizer.json " + dst_path)
    os.system("cp -rf models/configuration_xmodel2.py " + dst_path)
    os.system("cp -rf models/modeling_xmodel2.py " + dst_path)

    with open(os.path.join(dst_path, "config.json"), "r") as f:
        data = json.load(f)
        data["_attn_implementation"] = "flash_attention_2"

    with open(os.path.join(dst_path, "config.json"), "w") as f:
        json.dump(data, f, indent=2)

def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_path", type=str, default='/data2/liuyang/i_line_ckpt/i_line_s1_fp8_0921/')
    parser.add_argument("--dst_path", type=str, default='/data2/liuyang/i_line_ckpt/i_line_s1_fp8_0921-hf/')
    parser.add_argument("--reverse", action="store_true", help="reverse")

    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    src_path = args.src_path
    dst_path = args.dst_path

    ckpts = [ckp for ckp in os.listdir(src_path) if ckp.startswith('iter') and ckp.endswith('000')]
    ckpts = sorted(ckpts)
    # print(f'ckpts: {ckpts}')

    if args.reverse:
        ckpts = list(reversed(ckpts))

    for ckp in tqdm(ckpts):
        ckp_path = os.path.join(src_path, ckp)
        checkpoint_path = f'{ckp_path}/mp_rank_00/model_optim_rng.pt'
        save_dir = os.path.join(dst_path, ckp)

        try:
            convert_checkpoint(checkpoint_path, save_dir)
        except KeyError as err:
            print(ckp)
            print(err)
        except EOFError as err:
            print(ckp)
            print(err)
        except Exception as exp:
            print(ckp)
            print(err)