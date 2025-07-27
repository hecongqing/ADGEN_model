#!/bin/bash

# 启动vLLM推理服务脚本

set -e

echo "🚀 启动vLLM推理服务..."

# 默认参数
MODEL_PATH="${MODEL_PATH:-./outputs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "📋 服务配置:"
echo "  模型路径: ${MODEL_PATH}"
echo "  服务地址: ${HOST}:${PORT}"
echo "  工作进程: ${WORKERS}"
echo "  张量并行: ${TENSOR_PARALLEL_SIZE}"
echo "  GPU内存利用率: ${GPU_MEMORY_UTILIZATION}"
echo "  最大模型长度: ${MAX_MODEL_LEN}"
echo ""

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: ${MODEL_PATH}"
    echo "请先训练模型或指定正确的模型路径"
    exit 1
fi

# 构建启动命令
SERVICE_CMD="python service/vllm_service.py"
SERVICE_CMD="${SERVICE_CMD} --model_path ${MODEL_PATH}"
SERVICE_CMD="${SERVICE_CMD} --host ${HOST}"
SERVICE_CMD="${SERVICE_CMD} --port ${PORT}"
SERVICE_CMD="${SERVICE_CMD} --workers ${WORKERS}"
SERVICE_CMD="${SERVICE_CMD} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
SERVICE_CMD="${SERVICE_CMD} --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION}"
SERVICE_CMD="${SERVICE_CMD} --max_model_len ${MAX_MODEL_LEN}"

echo "🔧 执行启动命令:"
echo "${SERVICE_CMD}"
echo ""

# 执行启动命令
eval ${SERVICE_CMD}