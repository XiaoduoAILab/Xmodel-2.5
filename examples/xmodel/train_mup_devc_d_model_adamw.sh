#!/bin/bash

# Runs the FP8 training script with Muon optimizer and Transformer Engine
# Combines techniques from both FP8 and Muon training scripts

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=out/mup_convert_debug_d_model_adamw
TENSORBOARD_LOGS_PATH=runs/mup_convert_debug_d_model_adamw
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
    --num-layers 12
    --hidden-size 512
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 2
    --ffn-hidden-size 1280
    --position-embedding-type rope
    --seq-length 2048
    --max-position-embeddings 131072
    --rotary-base 500000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.7684243289772825
    --attention-backend fused
    --use-flash-attn
    --normalization RMSNorm
    --disable-bias-linear
    --use-mup
    --mup-input-scale 0.009350692686916321
    --mup-output-scale 3.3813847646997552
    --mup-attention-residual-scale 0.03600486455598032
    --mup-ffn-residual-scale 0.0019072421409102565
)

TRAINING_ARGS=(
    --micro-batch-size 16
    --global-batch-size 32
    --train-iters 10000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr 0.004381083982757232   # decoupled-lr/512
    --decoupled-lr 2.243114999171703
    --lr-decay-style constant
    --lr-warmup-iters 0
    --bf16
    --cross-entropy-loss-fusion
    --no-decay-norm-bias-embed
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
    --num-workers 8
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-interval 2000
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

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
