RUN_NAME="xmodel2.5_long_context"
LANGUAGE_MODEL="/data2/liuyang/models/xmodel/llama/"
TOKENIZER_PATH="/data2/liuyang/Xmodel-2.5/tokenizers/deepseekv3/"
OUTPUT_DIR="./out/${RUN_NAME}/"

if [ -z "$3" ]
  then
    DATA_PATH="/data1/i_line_data/"
  else
    DATA_PATH="$3"
fi

CUDA_VISIBLE_DEVICES="$1" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=8 long_context/train_long_context.py \
    --model_name_or_path ${LANGUAGE_MODEL} \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --data_path "${DATA_PATH}" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 3000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 5 \
    --model_max_length 32768 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'XmodelDecoderLayer' \
    --tf32 True \
    --report_to tensorboard \
    --logging_dir "./runs/${RUN_NAME}/" \
    --gradient_checkpointing 1
