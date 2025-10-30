#!/usr/bin/env python3
# hpo_megatron.py - Hyperparameter Optimization for Megatron-LM using Bayesian methods
import argparse
import logging
import math
import os
import random
import subprocess
import threading
from pathlib import Path
import sys
import optuna
from optuna.samplers import TPESampler
import numpy as np
import torch
from transformers import AutoTokenizer

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from models.configuration_xmodel2 import XmodelConfig
from models.modeling_xmodel2 import XmodelForCausalLM


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rms1_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('rms1-search')


class MegatronHPO:
    def __init__(self, output_dir):
        """
        初始化Megatron超参优化器
        
        :param output_dir: 输出目录
        """
        self.output_dir = Path(output_dir).resolve()

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")


    def _build_model(self, trial):
        """
        构建Megatron训练命令
        
        :param trial: Optuna试验对象
        :param trial_dir: 试验输出目录
        :return: 训练命令列表
        """
        # 超参数采样
        params = {
            "use_mup": trial.suggest_categorical('use_mup', [True, False]),
            "init_std": trial.suggest_float("init_std", 0.01, 0.1),
            "mup_input_scale": trial.suggest_float("mup_input_scale", 0, 20),
            "mup_output_scale": trial.suggest_float("mup_output_scale", 0, 20),
            "mup_attention_residual_scale": trial.suggest_float("mup_attention_residual_scale", 0.001, 1000),
            "mup_ffn_residual_scale": trial.suggest_float("mup_ffn_residual_scale", 0.01, 100)
        }

        use_mup = params["use_mup"]
        # use_mup = False
        init_std = params["init_std"]
        mup_input_scale = params["mup_input_scale"]
        mup_output_scale = params["mup_output_scale"]
        mup_attention_residual_scale = params["mup_attention_residual_scale"]
        mup_ffn_residual_scale = params["mup_ffn_residual_scale"]

        config = XmodelConfig.from_name(args.model)
        config.vocab_size = args.vocab_size
        config.torch_dtype = "bfloat16"

        config.hidden_size = args.hidden_size
        config.num_layers = args.num_layers
        config.num_attention_heads = args.hidden_size // 64
        config.ffn_hidden_size = args.hidden_size * 5 // 2
        config.num_key_value_heads = args.num_key_value_heads

        config.initializer_range = init_std
        config.use_mup = use_mup
        config.mup_input_scale = mup_input_scale
        config.mup_output_scale = mup_output_scale
        config.mup_attention_residual_scale = mup_attention_residual_scale
        config.mup_ffn_residual_scale = mup_ffn_residual_scale
        config._attn_implementation = "eager"

        model = XmodelForCausalLM(config)
        
        return model, params


    def _run_trial(self, trial):
        """
        运行单个超参数试验
        
        :param trial: Optuna试验对象
        :return: 验证损失
        """

        device = "cpu"  # 使用CPU进行激活收集

        # 构建训练命令
        model, params = self._build_model(trial)
        model = model.to(device)
        logger.info(f"Starting trial {trial.number} with params: {params}")
    

        # 确保tokenizer有pad_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 使用字典存储各层激活值，避免hook冲突
        activation_data = {
            'attn_layers': [],
            'ffn_layers': []
        }

        # 定义hook函数收集各层激活值
        def get_activation(layer_type, layer_idx):
            def hook(model, input, output):
                # 处理元组输出（transformer层常见）
                if isinstance(output, tuple):
                    activation = output[0].detach()
                else:
                    activation = output.detach()
                
                # 计算RMS值（跨所有空间维度）
                rms = torch.sqrt(torch.mean(activation ** 2))
                
                if layer_type == 'attn':
                    activation_data['attn_layers'][layer_idx] = rms.item()
                else:
                    activation_data['ffn_layers'][layer_idx] = rms.item()
            return hook

        # 初始化激活值存储
        activation_data['attn_layers'] = [0.0] * len(model.model.layers)
        activation_data['ffn_layers'] = [0.0] * len(model.model.layers)

        # 注册hook
        hooks = []
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.register_forward_hook(get_activation('attn', i)))
            hooks.append(layer.mlp.register_forward_hook(get_activation('ffn', i)))

        # 准备输入数据（固定长度的序列）
        input_ids = torch.randint(0, args.vocab_size, (1, args.n_tokens), device=device)
        attention_mask = torch.ones_like(input_ids)

        # 运行单次前向传播
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 收集embedding和logits的RMS
            embeddings = model.model.embed_tokens(input_ids)
            if model.config.use_mup:
                embeddings *= model.config.mup_input_scale
            embedding_rms = torch.sqrt(torch.mean(embeddings ** 2)).item()
            
            logits = outputs.logits
            logits_rms = torch.sqrt(torch.mean(logits ** 2)).item()

        # 移除hook
        for hook in hooks:
            hook.remove()

        # 构建激活值字典
        activations = {
            'word_embedding': embedding_rms,
            'attention_outputs': activation_data['attn_layers'],
            'ffn_outputs': activation_data['ffn_layers'],
            'output_logits': logits_rms
        }


        # 计算每层激活值的RMS（Root Mean Square）
        def compute_activation_stats(activations):
            stats = {}
            for name, values in activations.items():
                if name.endswith('outputs'):
                    stats[name] = values  # 已经是RMS值
                else:
                    stats[name] = values  # 已经是RMS值
            return stats
        
        stats = compute_activation_stats(activations)

        dist = 0.0
        for attn_rms in stats['attention_outputs']:
            dist += (attn_rms - 1.0) ** 2
        for ffn_rms in stats['ffn_outputs']:
            dist += (ffn_rms - 1.0) ** 2

        if args.include_embed:
            dist += (stats['word_embedding'] - 1.0) ** 2
            dist += (stats['output_logits'] - 1.0) ** 2

        dist = math.sqrt(dist)

        return dist

    def optimize(self, n_trials=50, timeout=None, n_jobs=1):
        """
        运行超参数优化
        
        :param n_trials: 最大试验次数
        :param timeout: 最大运行时间（秒）
        :param n_jobs: 并行作业数 (最大8，每个试验使用1个GPU)
        """
        # 限制最大并行数不超过8
        n_jobs = min(n_jobs, 8)
        # 创建Optuna study
        sampler = TPESampler(n_startup_trials=10, multivariate=True)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name="optuna_rms1_search"
        )

        # 运行优化
        study.optimize(
            self._run_trial,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        # 输出最佳结果
        logger.info("\n===== Optimization Completed =====")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best RMS distance: {study.best_trial.value:.4f}")
        logger.info("Best parameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # 保存结果
        results_file = self.output_dir / "rms1_search_results.csv"
        study.trials_dataframe().to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Megatron-LM")
    parser.add_argument("--tokenizer-path", default="tokenizers/deepseekv3", help="Path to tokenizer")
    parser.add_argument("--output-dir", default="rms1_search_results", help="Output directory for results")
    parser.add_argument("--trials", type=int, default=60, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument('--model', type=str, default='xl')
    parser.add_argument("--vocab_size", type=int, default=129280, help="Vocabulary size")
    parser.add_argument('--n_tokens', type=int, default=10, help='Number of tokens to average activations over')
    parser.add_argument("--include-embed", action="store_true", help="Include embedding layer in RMS calculation")
    parser.add_argument("--hidden_size", type=int, default=1536, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=48, help="Hidden size of the model")
    parser.add_argument("--num_key_value_heads", type=int, default=8, help="Number of key/value heads")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 初始化并运行优化器
    hpo = MegatronHPO(
        output_dir=args.output_dir,
    )

    hpo.optimize(
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.jobs
    )
