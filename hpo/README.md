# iterhpo  
**Iterative / Continual Hyper-Parameter Optimization for Megatron-LM**

---

## 1. 背景与动机  
在大模型预训练中，传统做法是**一次性在训练开始前选定学习率、batch size 等超参**，然后全程不变。  
然而，最优超参往往**随训练步数而变化**；如果在训练过程中**定期重新搜索并热启动 checkpoint**，就能让模型始终“跑在”当前最优的学习曲线上。  
这种策略在文献里常被称为

> **Continual / Iterative HPO with Warm-Start**  
> （简称 **iterhpo**）

---

## 2. 方法概览  

```
for k in 0 .. K-1:
    1. 以最新 checkpoint 为起点
    2. 在当前收缩的搜索空间里用贝叶斯优化重新选 (lr, batch_size)
    3. 继续训练 N 步
    4. 保存新 checkpoint 与本轮最优超参
```

- **每阶段步数 N**：  
  小模型 500–1 000，大模型 2 000–4 000；可用 pilot 实验把“能拉开 loss 差距的最小步数 ×1.5”作为最终 N。  
- **搜索算法**：  
  贝叶斯优化（Optuna + TPE），比网格搜索省 5–10× GPU 时。  
- **搜索空间收缩**：  
  每轮把上一轮最佳 (lr, bs) 作为中心，收缩 ±20 %/±50 %。  

---

## 3. 快速上手  

### 3.1 安装依赖  
```bash
pip install optuna
```

### 3.2 运行示例  

```bash
python iterhpo.py \
  --megatron-dir . \
  --data-path  /datasets/batch1_content_document \
  --output-dir ./iterhpo_out \
  --stages 5 \
  --steps-per-stage 2000 \
  --gpu-pool 0,1,2,3 \
  --gpus-per-trial 1 \
  --jobs 4
```

| 参数 | 含义 |
|------|------|
| `--stages` | 总阶段数 K |
| `--steps-per-stage` | 每阶段训练步数 N |
| `--gpu-pool` | 逗号分隔的 GPU id |
| `--gpus-per-trial` | 每个试验占用的 GPU 数量 |
| `--jobs` | 最大并行试验数 |

---

## 4. 目录结构  

```
iterhpo_out/
├── stage_0/
│   ├── trial_0/
│   ├── trial_1/
│   ├── ...
│   ├── best.json           # 本阶段最优超参与 loss
│   └── ckpt/               # 最优 checkpoint
├── stage_1/
│   └── ...
├── ...
└── hpo_summary.csv         # 所有试验汇总
```

---

## 5. 超参随步数变化的幂律现象  

实验发现，把各阶段最优 `(lr, bs)` 画在 log-log 图里，通常满足  

```
lr(t) ≈ α · t^(-β)  
bs(t) ≈ γ · t^(δ)
```

β≈0.5–1.0，δ≈0.5–0.8，与 OpenAI/Google 公开的大模型调度器一致。  
`iterhpo.py` 会自动把每阶段结果写入 `stage_*/best.json`，方便后续拟合 power-law。

---

## 6. 调优建议  

| 场景 | N 推荐值 | Trials/Stage | 备注 |
|---|---|---|---|
| 350 M–1 B | 500–1 000 | 8–12 | 单卡 < 2 h |
| 3 B–7 B | 1 000–2 000 | 12–15 | 4–8 卡 |
| 30 B–70 B | 2 000–4 000 | 15–20 | 32–64 卡 |

- **Early-stopping**：Optuna 的 `MedianPruner` 可在 loss 连续 3 次无改善时提前终止 trial。  
- **Warm-start**：每轮把上一轮所有 `(param, loss)` 喂给新的 TPE sampler，搜索空间越缩越小。

---

## 7. 许可证  
MIT © 2024 iterhpo contributors
```