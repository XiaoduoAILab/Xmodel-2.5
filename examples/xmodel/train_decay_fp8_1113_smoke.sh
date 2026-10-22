#!/bin/bash

# Smoke test: Load converted HF checkpoint and run 100 training steps
# to verify the checkpoint loads cleanly (no missing/unexpected keys).
#
# Usage:
#   cd /nas_train/app.e0031982/code/BaiZe-ISEDA2027/Xmodel-2.5-code
#   bash examples/xmodel/train_decay_fp8_1113_smoke.sh
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

# Load from converted checkpoint, save to a separate smoke dir
LOAD_PATH=out/hf_converted_532000
CHECKPOINT_PATH=out/smoke_test_532000
TENSORBOARD_LOGS_PATH=runs/smoke_test_532000
TOKENIZER_MODEL=tokenizers/deepseekv3
DATA_BASE="/nas_train/app.e0031982/code/BaiZe-ISEDA2027/data"
# 4-source blend (same weighting as v2): SFT 64% / UFW-L3 24% / Code 8% / Math 4%
# For smoke test: use first shard of each source only (enough for 100 steps)
DATA_PATH="0.64 ${DATA_BASE}/mix_sft_tok/mix_sft_train_s0 \
           0.24 ${DATA_BASE}/full_ufw_l3/ufw_l3_train_s0 \
           0.08 ${DATA_BASE}/full_ultradata_code_l3/code_l3_train_s0 \
           0.04 ${DATA_BASE}/full_ultradata_math_l3/math_l3_train_s0"


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

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 960
    --train-iters 100
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
    --lr-decay-iters 20000
    --lr-wsd-decay-iters 20000
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
    --log-interval 1
    --eval-interval 1000
    --save-interval 1000
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

echo "=========================================="
echo "Smoke test: Loading from $LOAD_PATH"
echo "Training 100 steps, saving to $CHECKPOINT_PATH"
echo "=========================================="

/nas_train/app.e0031982/miniforge3/envs/py310/bin/torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${FP8_ARGS[@]}
