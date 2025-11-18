#!/usr/bin/env python3
"""
LLM检查点评测脚本，通过YAML配置评测任务，自动管理评测进度。
"""
import argparse
import glob
import json
import os
import re
import subprocess
from typing import Dict, List

import yaml
from tqdm import tqdm


def load_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_checkpoints(checkpoint_paths: List[str]) -> List[str]:
    """查找所有匹配的模型检查点路径"""
    checkpoints = set()
    for pattern in checkpoint_paths:
        # 处理目录路径（直接添加）
        if os.path.isdir(pattern):
            checkpoints.add(pattern)
        # 处理通配符路径
        elif '*' in pattern:
            checkpoints = checkpoints.union(set(glob.glob(pattern)))

    return sorted(checkpoints)


def get_result_path(checkpoint: str) -> str:
    """获取结果文件路径"""
    if checkpoint[-1] == '/':
        checkpoint = checkpoint[:-1]
    dir_name = os.path.basename(checkpoint)
    return os.path.join(os.path.dirname(__file__), f"results/{dir_name}_results.json")


def load_results(result_path: str) -> Dict:
    """加载结果文件"""
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            try:
                return json.load(f)
            except json.decoder.JSONDecodeError:
                return dict()
    return dict()


def run_lm_eval(checkpoint: str, task_name: str, task: Dict) -> Dict:
    """运行lm_eval评测并返回结果"""

    command = f"lm_eval --model hf --model_args pretrained={checkpoint},trust_remote_code=True,enable_thinking=False --tasks {task_name} --device cuda:{args.device} --batch_size {task['batch_size']} --num_fewshot {task['num_fewshot']} --confirm_run_unsafe_code "
    print('command: ' + str(command))

    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    print('stdout: ' + str(result.stdout))

    matches = re.findall(f"\|{task['metric']}\s*\|↑*\s*\|\s*([\d.]+)\s*\|", result.stdout, re.DOTALL)
    match = matches[0] if matches else None
    acc = float(match)
    print(f'result: {acc}')
    return acc


def parse_lm_eval_output(output: str) -> Dict:
    """解析lm_eval的输出表格"""
    results = {}
    # 示例输出格式:
    # |    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
    # |-------------|------:|------|-----:|--------|---|-----:|---|-----:|
    # |arc_challenge|      1|none  |    25|acc     |↑  |0.4130|±  |0.0144|
    # |             |       |none  |    25|acc_norm|↑  |0.4531|±  |0.0145|
    # |arc_easy     |      1|none  |    25|acc     |↑  |0.7441|±  |0.0090|
    # |             |       |none  |    25|acc_norm|↑  |0.7521|±  |0.0089|
    lines = output.split('\n')
    for line in lines:
        if '|' not in line or line.startswith('| Task') or line.startswith('|---'):
            continue

        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 4:
            task = parts[0]
            metric = parts[2]
            value = float(parts[3])
            if task not in results:
                results[task] = {}
            results[task][metric] = value
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = find_checkpoints(config["checkpoint_paths"])
    print(f"Found checkpoints: {checkpoints}")

    for checkpoint in tqdm(checkpoints):
        result_path = get_result_path(checkpoint)
        results = load_results(result_path)

        for task_name in config["tasks"].keys():
            if task_name not in results:
                task = config["tasks"][task_name]
                try:
                    results[task_name] = run_lm_eval(checkpoint, task_name, task)
                except KeyboardInterrupt:
                    raise
                except:
                    pass  # 如果评测失败，跳过该任务

        with open(result_path, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint directory to evaluate")
    parser.add_argument("--device", type=str, default="3", help="Device to run the evaluation on, e.g., '0'")
    args = parser.parse_args()

    main()
