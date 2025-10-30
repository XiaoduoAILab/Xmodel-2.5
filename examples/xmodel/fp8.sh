#!/bin/bash

# Runs the FP8 training script with Transformer Engine
# See more details at: https://github.com/NVIDIA/TransformerEngine

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=out/fp8
TENSORBOARD_LOGS_PATH=runs/megatron_lm_1b_fp8
TOKENIZER_MODEL=tokenizers/deepseekv3
DATA_PATH=/data1/i_line_data/ultrafineweb-en_batch1/batch1_content_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --use-mcore-models
    --num-layers 48
    --hidden-size 1536
    --num-attention-heads 24
    --group-query-attention
    --num-query-groups 8
    --ffn-hidden-size 3840
    --position-embedding-type rope
    --seq-length 4096
    --max-position-embeddings 131072
    --rotary-base 500000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.006
    --attention-backend fused
    --normalization RMSNorm
    --disable-bias-linear    
)

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 240
    --train-iters 260000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr 0.001667
    --lr-decay-style constant
    --lr-warmup-iters 2000
    --bf16
    --cross-entropy-loss-fusion
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
    --tokenizer-model $TOKENIZER_MODEL
    --tokenizer-type HuggingFaceTokenizer
    --vocab-size 129280
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-interval 200
    --save-interval 2000
    --log-params-norm
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

FP8_ARGS=(
    --transformer-impl "transformer_engine"
    --fp8-format hybrid
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max   
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${FP8_ARGS[@]}
