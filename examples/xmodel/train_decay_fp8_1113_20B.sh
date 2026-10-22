#!/bin/bash

# Full 20B-token decay training: WSD exponential, FP8, muon optimizer
# Starting from step_532000 (HF→Megatron converted checkpoint)
# Based on official train_decay_fp8_1113.sh
#
# Usage:
#   cd /nas_train/app.e0031982/code/BaiZe-ISEDA2027/Xmodel-2.5-code
#   bash examples/xmodel/train_decay_fp8_1113_20B.sh
#
# Prerequisites:
#   - Converted checkpoint at out/hf_converted_532000/
#   - Run tools/checkpoint/hf_to_megatron_mup.py first

# ========== 环境变量优化 ==========
export PATH=/nas_train/app.e0031982/miniforge3/envs/py310/bin:$PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions


GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=out/decay_20B_fp8
TENSORBOARD_LOGS_PATH=runs/decay_20B_fp8
TOKENIZER_MODEL=tokenizers/deepseekv3
LOAD_PATH=out/hf_converted_532000

# 4-source blend: SFT 64% / UFW-L3 24% / Code 8% / Math 4%
DATA_PATH="0.64 /nas_train/app.e0031982/code/BaiZe-ISEDA2027/data/mix_sft_tok/mix_sft_train_s0 \
           0.24 /nas_train/app.e0031982/code/BaiZe-ISEDA2027/data/full_ufw_l3/ufw_l3_train_s0 \
           0.08 /nas_train/app.e0031982/code/BaiZe-ISEDA2027/data/full_ultradata_code_l3/code_l3_train_s0 \
           0.04 /nas_train/app.e0031982/code/BaiZe-ISEDA2027/data/full_ultradata_math_l3/math_l3_train_s0"

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
    --seq-length 3776
    --max-position-embeddings 131072
    --rotary-base 500000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.1
    --attention-backend fused
    --use-flash-attn
    --normalization RMSNorm
    --disable-bias-linear
    --use-mup
    --mup-input-scale 12.0
    --mup-output-scale 1.0
    --mup-attention-residual-scale 1.4
    --mup-ffn-residual-scale 1.4
)

# 20B tokens / (960 * 3776) ≈ 5560 iterations
# WSD exponential decay over full 5560 iters (warmup=0, start decay immediately)
TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 960
    --train-iters 5560
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr 1.66667e-3
    --decoupled-lr 1e-2
    --lr-decay-style WSD
    --lr-wsd-decay-style exponential
    --min-lr 1.041667e-4
    --decoupled-min-lr 6.25e-4
    --lr-decay-iters 5560
    --lr-wsd-decay-iters 5560
    --lr-warmup-iters 0
    --bf16
    --cross-entropy-loss-fusion
    --no-decay-norm-bias
    --no-load-optim
    --finetune
    --optimizer muon
    --muon-matched-adamw-rms 0.2
)

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
    --make-vocab-size-divisible-by 129408
    --num-workers 8
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-interval 1000
    --save-interval 500
    --log-params-norm
    --log-throughput
    --ckpt-format torch
    --save $CHECKPOINT_PATH
    --load $LOAD_PATH
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --exit-on-missing-checkpoint
)

FP8_ARGS=(
    --transformer-impl "transformer_engine"
    --fp8-format hybrid
    --fp8-amax-history-len 128
    --fp8-amax-compute-algo max
)

PYTORCH_TENSOR_ALLOC_CONF=expandable_segments:True \
/nas_train/app.e0031982/miniforge3/envs/py310/bin/torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${FP8_ARGS[@]}
