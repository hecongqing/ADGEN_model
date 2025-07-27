#!/bin/bash

# Qwen广告文案生成模型训练脚本

set -e

echo "🚀 开始训练Qwen广告文案生成模型..."

# 默认参数
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
DATASET_PATH="${DATASET_PATH:-}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
USE_LORA="${USE_LORA:-true}"
USE_8BIT="${USE_8BIT:-true}"

echo "📋 训练配置:"
echo "  模型: ${MODEL_NAME}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  数据集路径: ${DATASET_PATH:-从ModelScope自动下载}"
echo "  训练轮数: ${EPOCHS}"
echo "  批大小: ${BATCH_SIZE}"
echo "  学习率: ${LEARNING_RATE}"
echo "  最大长度: ${MAX_LENGTH}"
echo "  使用LoRA: ${USE_LORA}"
echo "  使用8bit: ${USE_8BIT}"
echo ""

# 构建训练命令
TRAIN_CMD="python -m src.train"
TRAIN_CMD="${TRAIN_CMD} --model_name ${MODEL_NAME}"
TRAIN_CMD="${TRAIN_CMD} --output_dir ${OUTPUT_DIR}"
TRAIN_CMD="${TRAIN_CMD} --num_train_epochs ${EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --per_device_train_batch_size ${BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --learning_rate ${LEARNING_RATE}"
TRAIN_CMD="${TRAIN_CMD} --max_length ${MAX_LENGTH}"

if [ ! -z "$DATASET_PATH" ]; then
    TRAIN_CMD="${TRAIN_CMD} --dataset_path ${DATASET_PATH}"
fi

if [ "$USE_LORA" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_lora"
fi

if [ "$USE_8BIT" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_8bit"
fi

TRAIN_CMD="${TRAIN_CMD} --test_after_train"

echo "🔧 执行训练命令:"
echo "${TRAIN_CMD}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 执行训练
eval ${TRAIN_CMD}

echo ""
echo "✅ 训练完成！"
echo "📁 模型保存在: ${OUTPUT_DIR}"
echo "📊 查看训练日志: tensorboard --logdir ${OUTPUT_DIR}/logs"