# Qwen广告文案生成器

基于Qwen3大语言模型的智能广告文案生成系统，支持LoRA微调、vLLM高性能推理和可视化界面。

## 🚀 功能特点

- 🎯 **基于Qwen3模型**: 使用先进的Qwen2.5-7B-Instruct模型
- 🚀 **LoRA微调**: 高效的参数微调，支持8bit量化
- 📊 **AdvertiseGen数据集**: 自动加载ModelScope上的广告文案数据集
- 🌐 **vLLM推理服务**: 高性能模型推理服务
- 🎨 **Gradio界面**: 友好的Web可视化界面
- 🔄 **批量生成**: 支持单个和批量文案生成
- 📈 **可调参数**: 支持温度、top-p等生成参数调节

## 📁 项目结构

```
├── dataset/                # 数据集缓存目录
├── outputs/                # 模型输出目录
├── configs/                # 配置文件
│   └── training_config.yaml
├── scripts/                # 启动脚本
│   ├── train.sh           # 训练脚本
│   ├── start_service.sh   # 启动推理服务
│   └── start_app.sh       # 启动应用界面
├── service/               # vLLM推理服务
│   ├── __init__.py
│   └── vllm_service.py
├── src/                   # 源代码
│   ├── __init__.py
│   ├── app.py            # Gradio应用界面
│   ├── dataset.py        # 数据集处理
│   ├── model.py          # 模型定义和推理
│   ├── train.py          # 训练脚本
│   └── utils.py          # 工具函数
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明
```

## 🛠️ 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 如果使用GPU，确保安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 使用说明

### 1. 模型训练

#### 方法1: 使用脚本（推荐）

```bash
# 使用默认参数训练
./scripts/train.sh

# 自定义参数训练
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
OUTPUT_DIR="./my_model" \
EPOCHS=5 \
BATCH_SIZE=8 \
./scripts/train.sh
```

#### 方法2: 直接运行Python

```bash
# 基础训练
python -m src.train \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "./outputs" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --use_lora \
    --use_8bit \
    --test_after_train

# 使用本地数据集
python -m src.train \
    --dataset_path "./my_dataset.json" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "./outputs"
```

### 2. 启动推理服务

#### 方法1: 使用脚本（推荐）

```bash
# 使用默认参数启动
./scripts/start_service.sh

# 自定义参数启动
MODEL_PATH="./outputs" \
PORT=8080 \
TENSOR_PARALLEL_SIZE=2 \
./scripts/start_service.sh
```

#### 方法2: 直接运行Python

```bash
python service/vllm_service.py \
    --model_path "./outputs" \
    --host "0.0.0.0" \
    --port 8000 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9
```

### 3. 启动Web应用

#### 方法1: 使用脚本（推荐）

```bash
# 使用本地模型
./scripts/start_app.sh

# 使用远程服务
USE_LOCAL_MODEL=false \
SERVICE_URL="http://localhost:8000" \
./scripts/start_app.sh

# 创建公共链接
SHARE=true ./scripts/start_app.sh
```

#### 方法2: 直接运行Python

```bash
# 使用本地模型
python -m src.app \
    --model_path "./outputs" \
    --use_local_model \
    --server_port 7860

# 使用远程服务
python -m src.app \
    --service_url "http://localhost:8000" \
    --server_port 7860
```

## 🎯 API使用

### 推理服务API

启动推理服务后，可以通过HTTP API调用：

```python
import requests

# 单个生成
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "类型#裙*版型#显瘦*材质#网纱*风格#性感*图案#蕾丝",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
})

result = response.json()
print(result["generated_text"])

# 批量生成
response = requests.post("http://localhost:8000/batch_generate", json={
    "prompts": [
        "类型#裙*版型#显瘦*材质#网纱*风格#性感",
        "类型#裤*版型#宽松*材质#牛仔*颜色#蓝色"
    ],
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
})

results = response.json()
for result in results:
    print(result["generated_text"])
```

### 本地模型使用

```python
from src.model import QwenAdvertiseGenerator

# 初始化生成器
generator = QwenAdvertiseGenerator()

# 加载预训练模型
generator.load_model()

# 或加载微调模型
# generator.load_finetuned_model("./outputs")

# 生成文案
content = "类型#裙*版型#显瘦*材质#网纱*风格#性感*图案#蕾丝"
result = generator.generate_text(
    content,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)
print(result)
```

## 📊 数据格式

### 输入格式

广告文案生成的输入格式为：`属性#值*属性#值*...`

常用属性包括：
- **类型**: 裙、裤、上衣、鞋、包等
- **版型**: 显瘦、宽松、修身、直筒等
- **材质**: 网纱、牛仔、棉、皮革、帆布等
- **风格**: 性感、休闲、简约、商务、文艺等
- **颜色**: 蓝色、白色、黑色、米色等
- **图案**: 蕾丝、镂空、纯色、格子等

### 示例

```
输入: 类型#裙*版型#显瘦*材质#网纱*风格#性感*图案#蕾丝*图案#镂空*图案#纯色*裙下摆#鱼尾*裙长#连衣裙
输出: 性感镂空蕾丝网纱鱼尾连衣裙，显瘦版型，纯色设计，优雅迷人。

输入: 类型#裤*版型#宽松*材质#牛仔*颜色#蓝色*风格#休闲*款式#直筒
输出: 蓝色宽松直筒牛仔裤，休闲舒适，百搭时尚单品。
```

## 🔧 配置说明

主要配置文件 `configs/training_config.yaml`:

```yaml
# 数据集配置
dataset:
  name: "AdvertiseGen"
  source: "modelscope"
  max_length: 512

# 模型配置
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  use_lora: true
  use_8bit: true

# LoRA配置
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1

# 训练配置
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 5.0e-5
  
# 生成配置
generation:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
```

## 📈 性能优化

### GPU内存优化

1. **8bit量化**: 减少约50%的GPU内存使用
2. **LoRA微调**: 只训练少量参数，大幅减少内存需求
3. **梯度累积**: 通过`gradient_accumulation_steps`模拟更大的batch size

### 推理优化

1. **vLLM加速**: 使用vLLM进行高性能推理
2. **批量处理**: 支持批量生成提高吞吐量
3. **张量并行**: 多GPU环境下使用`tensor_parallel_size`

## 🐛 常见问题

### 1. CUDA内存不足

```bash
# 减小batch size
--per_device_train_batch_size 2

# 使用梯度累积
--gradient_accumulation_steps 2

# 启用8bit量化
--use_8bit
```

### 2. 模型加载失败

```bash
# 检查模型路径
ls -la ./outputs/

# 检查模型文件
ls -la ./outputs/pytorch_model.bin

# 重新下载模型
rm -rf ./dataset/models--Qwen--Qwen2.5-7B-Instruct
```

### 3. vLLM服务启动失败

```bash
# 检查GPU可用性
nvidia-smi

# 降低GPU内存利用率
--gpu_memory_utilization 0.7

# 使用CPU模式（开发测试）
export CUDA_VISIBLE_DEVICES=""
```

## 📝 开发说明

### 添加新的数据处理功能

修改 `src/dataset.py` 中的 `AdvertiseGenDataset` 类：

```python
def custom_preprocess(self, examples):
    # 自定义预处理逻辑
    pass
```

### 自定义模型配置

修改 `src/model.py` 中的 `QwenAdvertiseGenerator` 类：

```python
def _setup_generation_config(self):
    # 自定义生成配置
    pass
```

### 添加新的评估指标

修改 `src/utils.py` 中的 `calculate_metrics` 函数：

```python
def calculate_metrics(predictions, references):
    # 添加新的评估指标
    pass
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 基础语言模型
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [Gradio](https://github.com/gradio-app/gradio) - Web界面框架
- [ModelScope](https://modelscope.cn/) - 数据集和模型仓库