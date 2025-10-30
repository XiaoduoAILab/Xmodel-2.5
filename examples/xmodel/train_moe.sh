#!/bin/bash
# DeepSeek-V3 精确 1.20 B total / 0.10 B activated  MoE + MLA
# Megatron-LM mcore 分支

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NVTE_FP8_DPA=1

# ------------------- 分布式 -------------------
GPUS_PER_NODE=8                # 按需修改
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$((GPUS_PER_NODE*NUM_NODES))

CHECKPOINT_PATH=out/dsv3_1.2b_exact
TENSORBOARD_LOGS_PATH=runs/dsv3_1.2b_exact
TOKENIZER_MODEL=tokenizers/deepseekv3
DATA_PATH="0.4100 /data1/i_line_data/ultrafineweb-en_content_document \
           0.2000 /data1/i_line_data/ultrafineweb-zh_content_document \
           0.2600 /data1/i_line_data/dolma_wo_cc/starcoder_text_document \
           0.0100 /data1/i_line_data/dolma_wo_cc/books_text_document \
           0.0050 /data1/i_line_data/dolma_wo_cc/algebraic-stack-train_text_document \
           0.0050 /data1/i_line_data/dolma_wo_cc/open-web-math-train_text_document \
           0.0100 /data1/i_line_data/dolma_wo_cc/wiki_text_document \
           0.0133 /data1/i_line_data/dolma_wo_cc/arxiv_text_document \
           0.0090 /data1/i_line_data/dolma_wo_cc/stackexchange_text_document \
           0.0391 /data1/i_line_data/dolma_wo_cc/reddit_text_document \
           0.0285 /data1/i_line_data/dolma_wo_cc/pes2o_text_document \
           0.0023 /data1/i_line_data/dolma_wo_cc/megawika_text_document \
           0.0078 /data1/i_line_data/dolma_wo_cc/tulu_flan_text_document"


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ---------- 1.20 B 精确模型 ----------
GPT_MODEL_ARGS=(
    --use-mcore-models

    # 主干
    --num-layers            14
    --hidden-size           640
    --ffn-hidden-size       1920        # 单专家 FFN 内维
    --num-attention-heads   10
    --seq-length            4096
    --max-position-embeddings 4096

    # MLA
    --multi-latent-attention
    --q-lora-rank           240
    --kv-lora-rank          160
    --qk-head-dim           128
    --v-head-dim            128

    # RoPE
    --position-embedding-type rope
    --rotary-base           10000
    --rotary-scaling-factor 40
    --rotary-percent        1.0

    # 正则化 / 激活
    --normalization         RMSNorm
    --swiglu
    --init-method-std       0.006
    --attention-dropout     0.0
    --hidden-dropout        0.0
    --disable-bias-linear
    --no-rope-fusion                    # 为了 MLA
    --attention-backend     fused
    --use-flash-attn

    # MoE（16 专家，无共享）
    --num-experts           16
    --moe-router-topk       4
    --moe-aux-loss-coeff    0.01
    --moe-expert-capacity-factor 1.0
    --moe-token-dispatcher-type alltoall
    --moe-router-pre-softmax

    # MTP-3（3 层额外预测）
    --mtp-num-layers 3
    --mtp-loss-scaling-factor 0.1
)

# ---------- 训练 ----------
TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 480
    --train-iters 270000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr 0.002
    --lr-decay-style constant
    --lr-warmup-iters 2000
    --bf16
    --recompute-activations
    --recompute-granularity selective
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --optimizer muon
    --muon-matched-adamw-rms 0.2
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
    --data-cache-path /data2/liuyang/meg_ds_cache/
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

# FP8
FP8_ARGS=(
    --transformer-impl "transformer_engine"
    --fp8-format hybrid
    --fp8-amax-history-len 32
    --fp8-amax-compute-algo max
)

# 启动
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${FP8_ARGS[@]}