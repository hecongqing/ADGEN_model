"""
工具函数模块
"""
import os
import json
import yaml
from typing import Dict, List, Any, Optional
from loguru import logger
import torch


def setup_logger(log_file: str = "app.log", level: str = "INFO"):
    """设置日志配置"""
    logger.remove()
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path}")


def save_config(config: Dict[str, Any], config_path: str):
    """保存配置文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, ensure_ascii=False, indent=2)
        elif config_path.endswith('.json'):
            json.dump(config, f, ensure_ascii=False, indent=2)


def get_device() -> torch.device:
    """获取设备信息"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
    return device


def format_prompt(content: str, instruction: str = None) -> str:
    """格式化提示词"""
    if instruction:
        return f"指令: {instruction}\n内容: {content}\n回答:"
    else:
        return f"根据以下关键词生成广告文案:\n{content}\n\n广告文案:"


def clean_generated_text(text: str) -> str:
    """清理生成的文本"""
    # 移除特殊token
    text = text.replace("<|endoftext|>", "").replace("<|pad|>", "")
    
    # 移除多余的空白字符
    text = text.strip()
    
    # 如果文本以常见的结束符结尾，保留；否则添加句号
    if not text.endswith(('。', '！', '？', '.', '!', '?')):
        if any(char in text for char in '。！？.!?'):
            pass  # 已有标点符号
        else:
            text += '。'
    
    return text


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算评估指标"""
    from collections import Counter
    import numpy as np
    
    def bleu_score(pred: str, ref: str) -> float:
        """简单的BLEU分数计算"""
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # 计算1-gram精确度
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        overlap = sum((pred_counter & ref_counter).values())
        precision = overlap / len(pred_tokens)
        
        return precision
    
    # 计算平均BLEU分数
    bleu_scores = [bleu_score(pred, ref) for pred, ref in zip(predictions, references)]
    avg_bleu = np.mean(bleu_scores)
    
    # 计算平均长度
    avg_pred_len = np.mean([len(pred.split()) for pred in predictions])
    avg_ref_len = np.mean([len(ref.split()) for ref in references])
    
    return {
        "bleu": avg_bleu,
        "avg_pred_length": avg_pred_len,
        "avg_ref_length": avg_ref_len,
        "length_ratio": avg_pred_len / avg_ref_len if avg_ref_len > 0 else 0.0
    }