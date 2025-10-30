#!/usr/bin/env python3
# hpo_megatron.py - Hyperparameter Optimization for Megatron-LM using Bayesian methods
import argparse
import logging
import os
import random
import subprocess
import threading
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
            "context_parallel_size": trial.suggest_categorical("context_parallel_size", [1]),
            "tensor_model_parallel_size": trial.suggest_categorical("tensor_model_parallel_size", [1]),
            "pipeline_model_parallel_size": trial.suggest_categorical("pipeline_model_parallel_size", [1]),
            "use_flash_attn": trial.suggest_categorical("use_flash_attn", [True, False]),
            "tp_comm_overlap": trial.suggest_categorical("tp_comm_overlap", [True, False]),
            "use_distributed_optimizer": trial.suggest_categorical("use_distributed_optimizer", [True, False]),
            "overlap_grad_reduce": trial.suggest_categorical("overlap_grad_reduce", [True, False]),
            "overlap_param_gather": trial.suggest_categorical("overlap_param_gather", [True, False]),
            "recompute_granularity": trial.suggest_categorical("recompute_granularity", ['full', 'selective']),
            "data_parallel_sharding_strategy": trial.suggest_categorical("data_parallel_sharding_strategy", ['no_shard', 'optim', 'optim_grads', 'optim_grads_params']),
            "recompute_modules": trial.suggest_categorical("recompute_modules", ['core_attn', 'core_attn mlp']),
        }

        # 生成10000到65535之间的随机整数
        random_port = random.randint(10000, 65535)

        weight_decay = 0.1
        # swiglu = True
        mup_base_depth = 4
        mup_base_width = 256
        hidden_size = 64
        num_layers = 4

        mup_width_multiplier = hidden_size / mup_base_width
        mup_depth_multiplier = num_layers / mup_base_depth
        num_attention_heads = hidden_size // 64
        ffn_hidden_size = hidden_size * 5 // 2

        train_iters = round(
            500 * mup_depth_multiplier * mup_width_multiplier)  # 调整训练迭代次数，使得 D=10N, 即数据量大约等于10倍参数量（嵌入层不算）。

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
            # "--no-rope-fusion",
            "--seq-length", "2048",
            "--max-position-embeddings", "131072",
            "--rotary-base", "500000",
            "--swiglu",
            "--init-method-std", "0.04",
            "--attention-backend", "fused",
            "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--micro-batch-size", "24",
            "--global-batch-size", "24",
            "--lr-decay-style", "constant",
            "--lr-warmup-iters", "0",
            "--bf16",
            "--train-iters", str(train_iters),
            "--clip-grad", "1.0",
            "--weight-decay", str(weight_decay),
            "--adam-beta1", "0.9",
            "--adam-beta2", "0.95",
            "--tensor-model-parallel-size", str(params["tensor_model_parallel_size"]),
            "--pipeline-model-parallel-size", str(params["pipeline_model_parallel_size"]),
            "--data-path", str(self.data_path),
            "--split", "999,1,0",
            "--tokenizer-model", "tokenizers/deepseekv3",
            "--tokenizer-type", "HuggingFaceTokenizer",
            "--vocab-size", "129280",
            "--eval-interval", str(train_iters),
            "--log-interval", "50",
            "--exit-on-missing-checkpoint",
            "--lr", "0.001",
            "--use-depth-mup",
            "--mup-input-scale", "12.0",
            "--mup-output-scale", "1.0",
            "--mup-attention-residual-scale", "2.2",
            "--mup-ffn-residual-scale", "1.2",
            "--no-decay-norm-bias-embed",
            "--context-parallel-size", str(params["context_parallel_size"]),
            "--recompute-granularity", str(params["recompute_granularity"]),
            "--data-parallel-sharding-strategy", str(params["data_parallel_sharding_strategy"]),
            "--recompute-modules", str(params["recompute_modules"]),
        ]

        if params["use_flash_attn"]:
            cmd.append("--use-flash-attn")

        if params["tp_comm_overlap"]:
            cmd.append("--tp-comm-overlap")

        if params["overlap_grad_reduce"]:
            cmd.append("--overlap-grad-reduce")

        if params["overlap_param_gather"]:
            cmd.append("--overlap-param-gather")

        if params["use_distributed_optimizer"]:
            cmd.append("--use-distributed-optimizer")

        return cmd, params

    def _parse_iteration_time(self, log_file: str) -> float:
        """
        从日志文件中解析每次迭代所需时间(毫秒)

        :param log_file: 日志文件路径
        :return: 平均每次迭代时间(ms)或inf
        """
        iteration_times = []

        try:
            log_text = open(log_file, 'r').read()
            lines = log_text.splitlines()

            for line in lines:
                if "elapsed time per iteration (ms):" in line:
                    parts = line.split('|')
                    for part in parts:
                        if "elapsed time per iteration (ms):" in part:
                            time_str = part.split(':')[1].strip()
                            iteration_times.append(float(time_str))

            if iteration_times:
                return sum(iteration_times) / len(iteration_times)
            return float('inf')

        except ValueError as ve:
            logger.error(f"Value error while parsing log file: {ve}")
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
        return float('inf')

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

                # 获取平均迭代时间
                avg_iter_time = self._parse_iteration_time(log_file)
                if avg_iter_time is None:
                    raise RuntimeError(f"Failed to parse iteration time for trial {trial.number}")

                return avg_iter_time

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
        :param n_jobs: 并行作业数 (最大8，每个试验使用1个GPU)
        """
        # 限制最大并行数不超过8
        n_jobs = min(n_jobs, 8)
        # 创建Optuna study
        sampler = TPESampler(n_startup_trials=10, multivariate=True)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name="hpo_mup_v7"
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
        logger.info(f"Best avg iteration time: {study.best_trial.value:.4f} ms")
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

    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "2"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

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
