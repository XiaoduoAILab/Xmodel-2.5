#!/usr/bin/env python3
"""
iterhpo.py  –  Iterative / Continual HPO for Megatron-LM
用法示例：
  python iterhpo.py --megatron-dir /opt/Megatron-LM \
                    --data-path /datasets/batch1_content_document \
                    --output-dir ./exp_7b_iterhpo \
                    --stages 5 --steps_per_stage 2000 \
                    --gpu-pool 0,1,2,3 --gpus-per-trial 1 --jobs 4
"""
import argparse, json, logging
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from hpo.run.hpo_megatron import MegatronHPO, GPUPoolManager   # 复用原脚本类

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("iterhpo")


class IterativeHPOTrainer:
    def __init__(
        self,
        megatron_dir: str,
        data_path: str,
        output_dir: str,
        gpus_per_trial: int = 1,
        gpu_pool=None,
        stages: int = 5,
        steps_per_stage: int = 2000,
    ):
        self.megatron_dir = Path(megatron_dir)
        self.data_path = Path(data_path)
        self.base_out = Path(output_dir)
        self.gpus_per_trial = gpus_per_trial
        self.gpu_pool = gpu_pool
        self.stages = stages
        self.steps_per_stage = steps_per_stage

        # 初始搜索空间
        self.lr0 = 0.006
        self.bs0 = 24
        self.lr_bounds = (self.lr0 / 4.0, self.lr0 * 4.0)
        self.bs_bounds = (max(6, self.bs0 // 4), self.bs0 * 4)

    # ------------------------------------------------------------
    # 工具：生成 stage 目录
    # ------------------------------------------------------------
    def stage_dir(self, k: int) -> Path:
        d = self.base_out / f"stage_{k}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------
    # 工具：根据上一阶段结果收缩搜索空间
    # ------------------------------------------------------------
    def _shrink_space(self, best_lr, best_bs, shrink=0.5):
        lr_low = max(self.lr_bounds[0], best_lr * (1 - shrink))
        lr_high = min(self.lr_bounds[1], best_lr * (1 + shrink))
        # Ensure batch sizes are multiples of 24 and at least 24
        bs_low = max(24, self.round_to_multiple(int(best_bs * (1 - shrink)), 24))
        bs_high = max(24, self.round_to_multiple(int(best_bs * (1 + shrink)), 24))
        return (lr_low, lr_high), (bs_low, bs_high)

    def round_to_multiple(self, value, multiple):
        """Round value to nearest multiple"""
        return multiple * round(value / multiple)

    # ------------------------------------------------------------
    # 核心：跑一个阶段 N 步 + 搜索
    # ------------------------------------------------------------
    def train_one_stage(self, stage: int, ckpt_in: Path = None):
        logger.info(f"========== STAGE {stage} ==========")
        out = self.stage_dir(stage)

        # 1. 准备当前阶段专用的 MegatronHPO
        hpo = MegatronHPO(
            megatron_dir=self.megatron_dir,
            data_path=self.data_path,
            output_dir=out,
            gpus_per_trial=self.gpus_per_trial,
            gpu_pool=self.gpu_pool,
        )

        # 2. 构造动态搜索空间
        if stage == 0:
            lr_lo, lr_hi = self.lr_bounds
            bs_lo, bs_hi = self.bs_bounds
        else:
            with open(out.parent / f"stage_{stage-1}" / "best.json") as f:
                prev = json.load(f)
            (lr_lo, lr_hi), (bs_lo, bs_hi) = self._shrink_space(
                prev["learning_rate"], prev["global_batch_size"]
            )

        # 3. 改写 hpo._build_train_command 以支持热启动
        original_build = hpo._build_train_command

        def patched_build(trial, trial_dir):
            cmd, params = original_build(trial, trial_dir)

            # 重写 train-iters 为当前阶段步数
            # new_cmd = []
            # i = 0
            # while i < len(cmd):
            #     if cmd[i] == "--train-iters":
            #         new_cmd.append(cmd[i])  # Keep the flag
            #         new_cmd.append(str(self.steps_per_stage))  # Replace the value
            #         i += 2  # Skip the original value
            #     else:
            #         new_cmd.append(cmd[i])
            #         i += 1
            # cmd = new_cmd

            # 若存在上一阶段 checkpoint，则热启动
            if ckpt_in and ckpt_in.exists():
                cmd += ["--load", str(ckpt_in)]

            # 写入本次实际搜索空间，便于后续收缩
            params.update({"stage": stage})
            return cmd, params

        hpo._build_train_command = patched_build

        # 4. 启动 Optuna 搜索
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(n_startup_trials=4, multivariate=True, seed=42),
        )

        def objective(trial):
            lr = trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True)
            bs = trial.suggest_int("global_batch_size", bs_lo, bs_hi, step=24)
            loss = hpo._run_trial(trial)
            return loss

        study.optimize(
            objective,
            n_trials=12,                    # 每阶段固定 12 个试验
            n_jobs=len(self.gpu_pool.available_gpus) // self.gpus_per_trial,
            show_progress_bar=True,
        )

        # 5. 保存本阶段最优结果
        best = study.best_trial.params
        best.update({"loss": study.best_value})
        with open(out / "best.json", "w") as f:
            json.dump(best, f, indent=2)
        logger.info(f"Stage {stage} best: {best}")

        # 6. 把最优 checkpoint 路径返回给下一阶段
        best_ckpt = out / f"trial_{study.best_trial.number}" / "ckpt"
        return best_ckpt

    # ------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------
    def run(self):
        ckpt = None
        for k in range(self.stages):
            ckpt = self.train_one_stage(k, ckpt_in=ckpt)
            if k < self.stages - 1:
                # 让下一阶段从最新 checkpoint 启动
                pass


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--megatron-dir", default=".")
    p.add_argument("--data-path", default="/datasets/batch1_content_document")
    p.add_argument("--output-dir", default="./iterhpo_out")
    p.add_argument("--stages", type=int, default=5, help="总阶段数 K")
    p.add_argument("--steps-per-stage", type=int, default=2000, help="每阶段 N 步")
    p.add_argument("--gpu-pool", default=None, help="0,1,2,3")
    p.add_argument("--gpus-per-trial", type=int, default=1)
    p.add_argument("--jobs", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.gpu_pool:
        pool = GPUPoolManager([int(g) for g in args.gpu_pool.split(",")])
    else:
        pool = GPUPoolManager(list(range(args.jobs)))

    trainer = IterativeHPOTrainer(
        megatron_dir=args.megatron_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        gpus_per_trial=args.gpus_per_trial,
        gpu_pool=pool,
        stages=args.stages,
        steps_per_stage=args.steps_per_stage,
    )
    trainer.run()
