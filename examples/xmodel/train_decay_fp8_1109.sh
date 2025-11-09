#!/bin/bash

# Runs the FP8 training script with AdamW optimizer and Transformer Engine
# Combines techniques from both FP8 and AdamW training scripts

# ========== 环境变量优化 ==========
export CUDA_DEVICE_MAX_CONNECTIONS=1
# NCCL优化参数
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_IB_DISABLE=1  # 如果使用InfiniBand可以移除或设置为0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions  # 避免/home空间不足


GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=out/i_line_s1_fp8_0921
TENSORBOARD_LOGS_PATH=runs/i_line_s1_fp8_0921
TOKENIZER_MODEL=tokenizers/deepseekv3
DATA_PATH="0.18725 /data1/i_line_data/ultrafineweb-en_content_document \
           0.03716 /data1/i_line_data/ultrafineweb-zh_content_document \
           0.02209 /data1/i_line_data/dolma_wo_cc/starcoder_text_document \
           0.00287 /data1/i_line_data/dolma_wo_cc/books_text_document \
           0.01722 /data1/i_line_data/dolma_wo_cc/algebraic-stack-train_text_document \
           0.01722 /data1/i_line_data/dolma_wo_cc/open-web-math-train_text_document \
           0.01147 /data1/i_line_data/dolma_wo_cc/wiki_text_document \
           0.00900 /data1/i_line_data/dolma_wo_cc/stackexchange_text_document \
           0.00900 /data1/i_line_data/dolma_wo_cc/reddit_text_document \
           0.01801 /data1/i_line_data/dolma_wo_cc/megawika_text_document \
           0.66870 /data1/i_line_data/sft_mixed_v2_deduped_v4_text-document"


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
    --seq-length 3712
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
    --micro-batch-size 5
    --global-batch-size 960
    --train-iters 550000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr 0.0016666666666666668   # decoupled-lr/(hidden_size/256)
    --decoupled-lr 0.01
    --lr-decay-style cosine
    --min-lr 0.00010416666666666667  # decoupled-min-lr/(hidden_size/256)
    --decoupled-min-lr 0.000625
    --lr-decay-iters 20000
    --lr-warmup-iters 0
    --bf16
    --cross-entropy-loss-fusion
    --no-decay-norm-bias
    --no-load-optim
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
    --eval-interval 1000
    --save-interval 1000
    --log-params-norm
    --log-throughput
    --ckpt-format torch
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

FP8_ARGS=(
    --transformer-impl "transformer_engine"
    --fp8-format hybrid
    --fp8-amax-history-len 128
    --fp8-amax-compute-algo max
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${FP8_ARGS[@]}
