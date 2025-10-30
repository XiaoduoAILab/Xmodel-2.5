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

import optuna
from optuna.samplers import GridSampler

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


class GPUPoolManager:
    """GPU资源池管理类"""

    def __init__(self, gpu_ids):
        self.available_gpus = list(gpu_ids)
        self.lock = threading.Lock()

    def acquire_gpu(self):
        """获取一个可用的GPU"""
        with self.lock:
            if not self.available_gpus:
                raise RuntimeError("No available GPUs in pool")
            gpu_id = self.available_gpus.pop(0)
            return gpu_id

    def release_gpu(self, gpu_id):
        """释放GPU回资源池"""
        with self.lock:
            if gpu_id not in self.available_gpus:
                self.available_gpus.append(gpu_id)
                self.available_gpus.sort()


class MegatronHPO:
    def __init__(self, megatron_dir, data_path, output_dir, gpus_per_trial=1, gpu_pool=None):
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
        self.gpu_pool = gpu_pool
        if gpu_pool is None:
            # 向后兼容模式
            self.gpu_pool = GPUPoolManager(range(0, args.jobs))

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
            "learning_rate": trial.suggest_categorical("learning_rate", [2 ** i for i in range(-15, -2)]),
            # "use_mup": trial.suggest_categorical('use_mup', [True, False]),
            # "hidden_size": trial.suggest_categorical("hidden_size", [256, 512, 768, 1024, 1280, 1536]),
            "use_mup": trial.suggest_categorical('use_mup', [True, False]),
            "hidden_size": trial.suggest_categorical("hidden_size", [256, 512, 1024, 2048]),
        }

        # 生成10000到65535之间的随机整数
        random_port = random.randint(10000, 65535)

        num_layers = 4
        mup_base_width = 256

        learning_rate = params["learning_rate"]
        hidden_size = params["hidden_size"]

        mup_width_multiplier = hidden_size // mup_base_width
        num_attention_heads = hidden_size // 64
        ffn_hidden_size = hidden_size * 5 // 2

        train_iters = 500 * mup_width_multiplier  # 根据hidden_size调整训练迭代次数，使得 D=10N, 即数据量大约等于10倍参数量（嵌入层不算）。

        # 构建命令
        cmd = [
            "torchrun",
            "--nproc_per_node", str(self.gpus_per_trial),
            "--nnodes", "1",
            "--master_addr", "localhost",
            "--master_port", f"{random_port}",  # 随机端口
            str(self.megatron_dir / "pretrain_gpt.py"),
            "--use-mcore-models",
            "--num-layers", str(num_layers),
            "--hidden-size", str(hidden_size),
            "--num-attention-heads", str(num_attention_heads),
            "--group-query-attention",
            "--num-query-groups", "1",
            "--ffn-hidden-size", str(ffn_hidden_size),
            "--position-embedding-type", "rope",
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
            "--lr", str(learning_rate),
            "--lr-decay-style", "constant",
            "--lr-warmup-iters", "0",
            "--bf16",
            "--train-iters", str(train_iters),
            "--clip-grad", "1.0",
            "--weight-decay", "0.1",
            "--adam-beta1", "0.9",
            "--adam-beta2", "0.95",
            "--tensor-model-parallel-size", "1",
            "--pipeline-model-parallel-size", "1",
            "--data-path", str(self.data_path),
            "--split", "999,1,0",
            "--tokenizer-model", "tokenizers/deepseekv3",
            "--tokenizer-type", "HuggingFaceTokenizer",
            "--vocab-size", "129280",
            "--eval-interval", str(train_iters),
            "--log-interval", "50",
            "--exit-on-missing-checkpoint",
            "--no-decay-norm-bias-embed",
            "--optimizer", "muon",
            "--muon-matched-adamw-rms", "0.2",
        ]

        # 根据参数添加mup相关选项
        if params["use_mup"]:
            cmd += ["--use-mup",
                    "--mup-input-scale", "1.0",
                    "--mup-output-scale", "1.0",
                    "--mup-attention-residual-scale", "1.0",
                    "--mup-ffn-residual-scale", "1.0",
                    ]

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
        :return: 训练集损失或inf
        """
        best_loss = float('inf')  # 默认值为无穷大

        try:
            log_text = open(log_file, 'r').read()
            # 查找包含训练集损失的行
            lines = log_text.splitlines()

            for line in lines:
                if "lm loss:" in line:
                    # 提取损失值
                    parts = line.split('|')
                    for part in parts:
                        if "lm loss:" in part:
                            loss_str = part.split(':')[1].strip()
                            if float(loss_str) < best_loss:
                                best_loss = float(loss_str)

        except ValueError as ve:
            logger.error(f"Value error while parsing log file: {ve}")
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
        return best_loss

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
            # 从GPU池获取GPU
            gpu_id = self.gpu_pool.acquire_gpu()
            logger.info(f"Acquired GPU {gpu_id} for trial {trial.number}")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            try:
                # 运行训练
                with open(log_file, 'w') as log:
                    process = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=self.megatron_dir,
                        preexec_fn=os.setsid,
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

            finally:
                # 确保GPU被释放
                self.gpu_pool.release_gpu(gpu_id)
                logger.info(f"Released GPU {gpu_id} from trial {trial.number}")

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # 返回一个较大的损失值表示失败
            return float('inf')

    def optimize(self, n_trials=50, timeout=None, n_jobs=1):
        """
        运行超参数优化

        :param n_trials: 最大试验次数
        :param timeout: 最大运行时间（秒）
        :param n_jobs: 并行作业数
        """
        # 限制最大并行数不超过8
        n_jobs = min(n_jobs, 8)

        search_space = {
            "learning_rate": [2 ** i for i in range(-15, -2)],
            "use_mup": [True, False],
            "hidden_size": [256, 512, 1024, 2048]
        }

        n_trials = len(search_space["learning_rate"]) * len(search_space["use_mup"]) * len(
            search_space["hidden_size"])

        # 创建Optuna study
        sampler = GridSampler(search_space=search_space)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name="muon_mup_width_v4"
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
    parser.add_argument("--gpu-pool", type=str, default=None,
                        help="Comma-separated list of GPU IDs to use as pool, e.g. '0,1,2,3'")
    parser.add_argument("--gpus-per-trial", type=int, default=1, help="GPUs per trial")
    parser.add_argument("--trials", type=int, default=60, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")

    args = parser.parse_args()

    # 初始化并运行优化器
    # 初始化GPU池
    if args.gpu_pool:
        gpu_pool = GPUPoolManager([int(gpu) for gpu in args.gpu_pool.split(',')])
    else:
        gpu_pool = None  # 使用向后兼容模式

    hpo = MegatronHPO(
        megatron_dir=args.megatron_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        gpus_per_trial=args.gpus_per_trial,
        gpu_pool=gpu_pool
    )

    hpo.optimize(
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.jobs
    )
