#!/bin/bash

# 启动Gradio应用脚本

set -e

echo "🎯 启动Gradio广告文案生成应用..."

# 默认参数
MODEL_PATH="${MODEL_PATH:-}"
SERVICE_URL="${SERVICE_URL:-http://localhost:8000}"
USE_LOCAL_MODEL="${USE_LOCAL_MODEL:-true}"
SERVER_NAME="${SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-7860}"
SHARE="${SHARE:-false}"

echo "📋 应用配置:"
echo "  模型路径: ${MODEL_PATH:-使用预训练模型}"
echo "  服务地址: ${SERVICE_URL}"
echo "  使用本地模型: ${USE_LOCAL_MODEL}"
echo "  应用地址: ${SERVER_NAME}:${SERVER_PORT}"
echo "  公共链接: ${SHARE}"
echo ""

# 构建启动命令
APP_CMD="python -m src.app"

if [ ! -z "$MODEL_PATH" ]; then
    APP_CMD="${APP_CMD} --model_path ${MODEL_PATH}"
fi

APP_CMD="${APP_CMD} --service_url ${SERVICE_URL}"
APP_CMD="${APP_CMD} --server_name ${SERVER_NAME}"
APP_CMD="${APP_CMD} --server_port ${SERVER_PORT}"

if [ "$USE_LOCAL_MODEL" = "true" ]; then
    APP_CMD="${APP_CMD} --use_local_model"
fi

if [ "$SHARE" = "true" ]; then
    APP_CMD="${APP_CMD} --share"
fi

echo "🔧 执行启动命令:"
echo "${APP_CMD}"
echo ""

# 执行启动命令
eval ${APP_CMD}