#!/usr/bin/env python3
# hpo_megatron.py - Hyperparameter Optimization for Megatron-LM using Bayesian methods

import argparse
import logging
import os
import random
import subprocess
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hpo_mup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mup-hpo')


class MegatronHPO:
    def __init__(self, megatron_dir, data_path, output_dir, gpus_per_trial=8):
        """
        初始化Megatron超参优化器
        
        :param megatron_dir: Megatron-LM源码目录
        :param data_path: 训练数据路径
        :param output_dir: 输出目录
        :param gpus_per_trial: 每个试验分配的GPU数量
        """
        self.megatron_dir = Path(megatron_dir).resolve()
        self.data_path = Path(data_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.gpus_per_trial = gpus_per_trial

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # 验证Megatron目录
        if not (self.megatron_dir / "pretrain_gpt.py").exists():
            raise FileNotFoundError("Megatron-LM directory is invalid")

    def _build_train_command(self, trial, trial_dir):
        """
        构建Megatron训练命令
        
        :param trial: Optuna试验对象
        :param trial_dir: 试验输出目录
        :return: 训练命令列表
        """
        # 超参数采样
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 0.1, log=True),
            "decoupled_lr": trial.suggest_float("decoupled_lr", 5e-4, 0.1, log=True),
            "use_mup": trial.suggest_categorical('use_mup', [True, False]),
            "use_depth_mup": trial.suggest_categorical('use_depth_mup', [True, False]),
            "mup_input_scale": trial.suggest_float("mup_input_scale", 1.0, 20.0),
            "mup_output_scale": trial.suggest_float("mup_output_scale", 0.5, 2.0),
            "mup_attention_residual_scale": trial.suggest_float("mup_attention_residual_scale", 1.0, 8.0),
            "mup_ffn_residual_scale": trial.suggest_float("mup_ffn_residual_scale", 1.0, 8.0)
        }

        # 生成10000到65535之间的随机整数
        random_port = random.randint(10000, 65535)

        # 构建命令
        cmd = [
            "torchrun",
            "--nproc_per_node", str(self.gpus_per_trial),
            "--nnodes", "1",
            "--master_addr", "localhost",
            "--master_port", f"{random_port}",  # 随机端口
            str(self.megatron_dir / "pretrain_gpt.py"),
            "--use-mcore-models",
            "--num-layers", "4",
            "--hidden-size", "256",
            "--num-attention-heads", "4",
            "--group-query-attention",
            "--num-query-groups", "1",
            "--ffn-hidden-size", "640",
            "--position-embedding-type", "rope",
            "--no-rope-fusion",
            "--seq-length", "2048",
            "--max-position-embeddings", "131072",
            "--rotary-base", "500000",
            "--swiglu",
            "--init-method-std", "0.02",
            "--attention-backend", "fused",
            "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--micro-batch-size", "12",
            "--global-batch-size", "24",
            "--lr", str(params["learning_rate"]),
            "--decoupled-lr", str(params["decoupled_lr"]),
            "--lr-decay-style", "constant",
            "--lr-warmup-iters", "0",
            "--bf16",
            "--train-iters", "1000",
            "--clip-grad", "1.0",
            "--weight-decay", "0.1",
            "--adam-beta1", "0.9",
            "--adam-beta2", "0.95",
            "--tensor-model-parallel-size", "1",
            "--pipeline-model-parallel-size", "1",
            "--data-path", "/datasets/batch1_content_document",
            "--split", "999,1,0",
            "--tokenizer-model", "tokenizers/deepseekv3",
            "--tokenizer-type", "HuggingFaceTokenizer",
            "--vocab-size", "129280",
            "--eval-interval", "999999999",  # 禁用评估
            "--log-interval", "50",
            "--exit-on-missing-checkpoint",
            "--mup-input-scale", str(params["mup_input_scale"]),
            "--mup-output-scale", str(params["mup_output_scale"]),
            "--mup-attention-residual-scale", str(params["mup_attention_residual_scale"]),
            "--mup-ffn-residual-scale", str(params["mup_ffn_residual_scale"]),
        ]

        # 根据参数添加mup相关选项
        if params["use_mup"]:
            cmd.append("--use-mup")
        if params["use_depth_mup"]:
            cmd.append("--use-depth-mup")

        # 添加分布式参数
        cmd += [
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--overlap-param-gather"
        ]

        return cmd, params

    def _parse_validation_loss(self, log_file: str) -> float:
        """
        从日志文件中解析损失
        
        :param log_file: 日志文件路径
        :return: lm损失值或None
        """
        try:
            log_text = open(log_file, 'r').read()
            # 查找包含损失的行
            lines = log_text.splitlines()
            last_loss = None
            for line in lines:
                if "lm loss:" in line:
                    # 提取损失值
                    parts = line.split('|')
                    for part in parts:
                        if "lm loss:" in part:
                            loss_str = part.split(':')[1].strip()
                            last_loss = float(loss_str)
            return last_loss
        except ValueError as ve:
            logger.error(f"Value error while parsing log file: {ve}")

        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
        return None

    def _run_trial(self, trial):
        """
        运行单个超参数试验
        
        :param trial: Optuna试验对象
        :return: 验证损失
        """
        # 创建试验目录
        trial_id = f"trial_{trial.number}"
        trial_dir = self.output_dir / trial_id
        trial_dir.mkdir(exist_ok=True)

        # 构建训练命令
        cmd, params = self._build_train_command(trial, trial_dir)
        logger.info(f"Starting trial {trial.number} with params: {params}")

        # 设置日志文件
        log_file = trial_dir / "train.log"

        try:
            # 为当前试验分配特定的GPU
            gpu_id = trial.number % 8  # 使用0-7号GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # 运行训练
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=self.megatron_dir,
                    preexec_fn=os.setsid,  # 创建新进程组
                    env=env
                )

            # 等待训练完成
            process.wait()

            # 获取最终损失
            final_lm_loss = self._parse_validation_loss(log_file)
            if final_lm_loss is None:
                raise RuntimeError(f"Failed to parse loss for trial {trial.number}")

            return final_lm_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # 返回一个较大的损失值表示失败
            return float('inf')

    def _get_current_step(self, log_file):
        """
        从日志文件中获取当前训练步数
        
        :param log_file: 日志文件路径
        :return: 当前步数
        """
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    if "iteration" in last_line:
                        parts = last_line.split()
                        for i, part in enumerate(parts):
                            if part == "iteration" and i + 1 < len(parts):
                                return int(parts[i + 1].strip('/'))
        except:
            pass
        return 0

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
            study_name="megatron_hpo"
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
        logger.info(f"Best lm loss: {study.best_trial.value:.4f}")
        logger.info("Best parameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # 保存结果
        results_file = self.output_dir / "hpo_results.csv"
        study.trials_dataframe().to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Megatron-LM")
    parser.add_argument("--megatron-dir", default=".", help="Path to Megatron-LM source directory")
    parser.add_argument("--data-path", default="/datasets/batch1_content_document", help="Path to training data")
    parser.add_argument("--output-dir", default="hpo_results", help="Output directory for results")
    parser.add_argument("--gpus-per-trial", type=int, default=1, help="GPUs per trial")
    parser.add_argument("--trials", type=int, default=60, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")

    args = parser.parse_args()

    # 初始化并运行优化器
    hpo = MegatronHPO(
        megatron_dir=args.megatron_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        gpus_per_trial=args.gpus_per_trial
    )

    hpo.optimize(
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.jobs
    )
