#!/bin/bash
# DeepSeek-V3 精确 20 M total / 1 B activated  MoE + MLA
# Megatron-LM mcore 分支

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=out/deepseekv3_60m_exact
TENSORBOARD_LOGS_PATH=runs/deepseekv3_60m_exact
TOKENIZER_MODEL=tokenizers/deepseekv3
DATA_PATH=/datasets/batch1_content_document


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --use-mcore-models

    # 主干 - 大幅减小规模以达到60M总参数
    --num-layers            6           # 进一步减少层数
    --hidden-size           256         # 大幅减小隐藏维度
    --ffn-hidden-size       512         # 大幅减小FFN内维
    --num-attention-heads   4           # 减少注意力头数
    --seq-length            1024        # 减小序列长度
    --max-position-embeddings 1024

    # MLA - 大幅减小LoRA秩
    --multi-latent-attention
    --q-lora-rank           64
    --kv-lora-rank          48
    --qk-head-dim           64
    --v-head-dim            64

    # RoPE
    --position-embedding-type rope
    --rotary-base           10000
    --rotary-scaling-factor 10
    --rotary-percent        1.0

    # 正则化 / 激活
    --normalization         RMSNorm
    --swiglu
    --init-method-std       0.005
    --attention-dropout     0.0
    --hidden-dropout        0.0
    --disable-bias-linear
    --no-rope-fusion
    --attention-backend     fused
    --use-flash-attn

    # MoE - 减少专家数量
    --num-experts           8           # 减少专家数
    --moe-router-topk       2           # 减少top-k
    --moe-aux-loss-coeff    0.01
    --moe-expert-capacity-factor 0.8    # 降低容量因子
    --moe-token-dispatcher-type alltoall
    --moe-router-pre-softmax

    # MTP-3
    --mtp-num-layers 2                  # 减少预测层数
    --mtp-loss-scaling-factor 0.05      # 调整损失缩放
)

# ---------- 训练 ----------
TRAINING_ARGS=(
    --micro-batch-size 16
    --global-batch-size 32
    --train-iters 200000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr 0.001
    --lr-decay-style constant
    --lr-warmup-iters 2000
    --bf16
    --cross-entropy-loss-fusion
    --no-decay-norm-bias
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# 并行
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
)

# 数据 & tokenizer
DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
    --tokenizer-model $TOKENIZER_MODEL
    --tokenizer-type HuggingFaceTokenizer
    --vocab-size 129280
)

# 日志 & 保存
EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-interval 2000
    --save-interval 2000
    --log-params-norm
    --log-throughput
    --ckpt-format torch
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
