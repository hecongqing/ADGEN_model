"""
模型训练脚本
"""
import os
import argparse
from typing import Optional
from loguru import logger

from .utils import setup_logger, get_device, save_config
from .dataset import AdvertiseGenDataset
from .model import QwenAdvertiseGenerator, QwenTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen广告文案生成模型训练")
    
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="数据集路径，默认从ModelScope加载")
    parser.add_argument("--cache_dir", type=str, default="./dataset",
                        help="缓存目录")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="预训练模型名称")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="是否使用LoRA微调")
    parser.add_argument("--use_8bit", action="store_true", default=True,
                        help="是否使用8bit量化")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每个设备的训练批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="每个设备的评估批大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="预热步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存间隔步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估间隔步数")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="日志记录间隔步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")
    
    # 其他参数
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--test_after_train", action="store_true",
                        help="训练后进行测试")
    parser.add_argument("--log_file", type=str, default="train.log",
                        help="日志文件路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()


def set_seed(seed: int = 42):
    """设置随机种子"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置日志
    setup_logger(args.log_file, level="INFO")
    logger.info("开始训练...")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备信息
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    try:
        # 1. 准备数据集
        logger.info("正在准备数据集...")
        dataset_manager = AdvertiseGenDataset(
            dataset_path=args.dataset_path,
            tokenizer_name=args.model_name,
            max_length=args.max_length,
            cache_dir=args.cache_dir
        )
        
        # 加载并预处理数据集
        processed_dataset = dataset_manager.prepare_dataset()
        train_dataset = processed_dataset['train']
        eval_dataset = processed_dataset['validation']
        
        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(eval_dataset)}")
        
        # 2. 初始化模型
        logger.info("正在初始化模型...")
        generator = QwenAdvertiseGenerator(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            device=device,
            use_lora=args.use_lora,
            use_8bit=args.use_8bit
        )
        
        # 加载模型
        model, tokenizer = generator.load_model()
        
        # 3. 创建训练器
        trainer_manager = QwenTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "logs")
        )
        
        # 4. 设置训练参数
        training_args = trainer_manager.get_training_arguments(
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        
        # 5. 开始训练
        trainer = trainer_manager.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            data_collator=dataset_manager.data_collator
        )
        
        # 6. 保存配置文件
        config = {
            "model_name": args.model_name,
            "dataset_path": args.dataset_path,
            "max_length": args.max_length,
            "use_lora": args.use_lora,
            "use_8bit": args.use_8bit,
            "training_args": training_args.to_dict(),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
        }
        
        config_path = os.path.join(args.output_dir, "training_config.json")
        save_config(config, config_path)
        logger.info(f"训练配置已保存到: {config_path}")
        
        # 7. 训练后测试
        if args.test_after_train:
            logger.info("开始训练后测试...")
            test_model(generator, dataset_manager)
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


def test_model(generator: QwenAdvertiseGenerator, dataset_manager: AdvertiseGenDataset):
    """测试训练后的模型"""
    logger.info("正在进行模型测试...")
    
    # 获取测试样本
    test_samples = dataset_manager.get_sample_data(num_samples=5)
    
    logger.info("测试样本生成结果:")
    logger.info("=" * 80)
    
    for i, sample in enumerate(test_samples, 1):
        content = sample['content']
        reference = sample['summary']
        
        # 生成广告文案
        try:
            generated = generator.generate_text(
                content,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9
            )
            
            logger.info(f"测试样本 {i}:")
            logger.info(f"输入: {content}")
            logger.info(f"参考: {reference}")
            logger.info(f"生成: {generated}")
            logger.info("-" * 40)
            
        except Exception as e:
            logger.error(f"生成测试样本 {i} 失败: {e}")
    
    logger.info("=" * 80)
    logger.info("模型测试完成")


def evaluate_model(model_path: str, dataset_path: Optional[str] = None):
    """评估模型性能"""
    logger.info(f"正在评估模型: {model_path}")
    
    # 初始化数据集
    dataset_manager = AdvertiseGenDataset(dataset_path=dataset_path)
    dataset_manager.load_from_modelscope()
    
    # 加载训练后的模型
    generator = QwenAdvertiseGenerator()
    generator.load_finetuned_model(model_path)
    
    # 获取评估数据
    eval_samples = dataset_manager.get_sample_data(num_samples=50)
    
    predictions = []
    references = []
    
    for sample in eval_samples:
        content = sample['content']
        reference = sample['summary']
        
        try:
            prediction = generator.generate_text(content)
            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            predictions.append("")
    
    # 计算评估指标
    from .utils import calculate_metrics
    metrics = calculate_metrics(predictions, references)
    
    logger.info("评估结果:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    main()