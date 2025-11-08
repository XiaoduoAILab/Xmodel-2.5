#!/bin/bash
# Llama-2 精确 67.6 M 参数版（无 MoE，无 MLA，无 MTP）
# 基于 Megatron-LM mcore 分支

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=out/llama2_67m
TENSORBOARD_LOGS_PATH=runs/llama2_67m
TOKENIZER_MODEL=tokenizers/deepseekv3    # tokenizer 不动
DATA_PATH=/datasets/batch1_content_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --use-mcore-models

    # ---------- Llama-2 主干 ----------
    --num-layers            10
    --hidden-size           384
    --ffn-hidden-size       1152
    --num-attention-heads   16
    --seq-length            2048
    --max-position-embeddings 2048

    # RoPE
    --position-embedding-type rope
    --rotary-base           10000
    --rotary-scaling-factor 20
    --rotary-percent        1.0

    # ---------- 正则化 / 激活 ----------
    --normalization         RMSNorm
    --swiglu
    --init-method-std       0.02
    --attention-dropout     0.0
    --hidden-dropout        0.0
    --disable-bias-linear
    --no-rope-fusion
    --attention-backend     fused
    --use-flash-attn
)

# ---------- 训练 ----------
TRAINING_ARGS=(
    --micro-batch-size 8
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