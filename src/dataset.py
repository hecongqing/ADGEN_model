"""
数据集处理模块
"""
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from loguru import logger

from .utils import setup_logger, format_prompt


class AdvertiseGenDataset:
    """广告文案生成数据集处理类"""
    
    def __init__(
        self, 
        dataset_path: Optional[str] = None,
        tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_length: int = 512,
        cache_dir: str = "./dataset"
    ):
        self.dataset_path = dataset_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.dataset = None
        
    def load_from_modelscope(self) -> DatasetDict:
        """从ModelScope加载AdvertiseGen数据集"""
        try:
            logger.info("正在从ModelScope加载AdvertiseGen数据集...")
            
            # 使用ModelScope API加载数据集
            from modelscope.msdatasets import MsDataset
            
            dataset = MsDataset.load(
                'lvjianjin/AdvertiseGen',
                cache_dir=self.cache_dir
            )
            
            # 转换为HuggingFace格式
            train_data = []
            dev_data = []
            
            # 处理训练集
            if 'train' in dataset:
                for item in dataset['train']:
                    train_data.append({
                        'content': item.get('content', ''),
                        'summary': item.get('summary', '')
                    })
            
            # 处理验证集
            if 'dev' in dataset:
                for item in dataset['dev']:
                    dev_data.append({
                        'content': item.get('content', ''),
                        'summary': item.get('summary', '')
                    })
            
            # 创建Dataset对象
            dataset_dict = DatasetDict({
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(dev_data) if dev_data else Dataset.from_list(train_data[:100])
            })
            
            self.dataset = dataset_dict
            logger.info(f"数据集加载完成: 训练集 {len(dataset_dict['train'])} 条, 验证集 {len(dataset_dict['validation'])} 条")
            
            return dataset_dict
            
        except Exception as e:
            logger.error(f"从ModelScope加载数据集失败: {e}")
            return self.load_from_local()
    
    def load_from_local(self) -> DatasetDict:
        """从本地文件加载数据集"""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            logger.warning("本地数据集路径不存在，创建示例数据集")
            return self.create_sample_dataset()
        
        try:
            logger.info(f"正在从本地加载数据集: {self.dataset_path}")
            
            # 根据文件格式选择加载方式
            if self.dataset_path.endswith('.json'):
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif self.dataset_path.endswith('.jsonl'):
                data = []
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            elif self.dataset_path.endswith('.csv'):
                df = pd.read_csv(self.dataset_path)
                data = df.to_dict('records')
            else:
                raise ValueError(f"不支持的文件格式: {self.dataset_path}")
            
            # 数据划分
            train_size = int(0.9 * len(data))
            train_data = data[:train_size]
            val_data = data[train_size:]
            
            dataset_dict = DatasetDict({
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(val_data)
            })
            
            self.dataset = dataset_dict
            logger.info(f"本地数据集加载完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")
            
            return dataset_dict
            
        except Exception as e:
            logger.error(f"从本地加载数据集失败: {e}")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self) -> DatasetDict:
        """创建示例数据集"""
        logger.info("创建示例数据集...")
        
        sample_data = [
            {
                "content": "类型#裙*版型#显瘦*材质#网纱*风格#性感*图案#蕾丝*图案#镂空*图案#纯色*裙下摆#鱼尾*裙长#连衣裙",
                "summary": "性感镂空蕾丝网纱鱼尾连衣裙，显瘦版型，纯色设计，优雅迷人。"
            },
            {
                "content": "类型#裤*版型#宽松*材质#牛仔*颜色#蓝色*风格#休闲*款式#直筒",
                "summary": "蓝色宽松直筒牛仔裤，休闲舒适，百搭时尚单品。"
            },
            {
                "content": "类型#上衣*版型#修身*材质#棉*颜色#白色*风格#简约*领型#圆领*袖长#长袖",
                "summary": "白色修身圆领长袖棉质上衣，简约设计，舒适百搭。"
            },
            {
                "content": "类型#鞋*材质#皮革*颜色#黑色*风格#商务*鞋跟#平底*款式#正装",
                "summary": "黑色皮革商务正装平底鞋，专业优雅，舒适耐穿。"
            },
            {
                "content": "类型#包*材质#帆布*颜色#米色*风格#文艺*容量#大容量*款式#单肩包",
                "summary": "米色帆布大容量单肩包，文艺清新，实用时尚。"
            }
        ] * 200  # 复制数据以增加样本量
        
        # 随机打乱数据
        import random
        random.shuffle(sample_data)
        
        # 数据划分
        train_size = int(0.8 * len(sample_data))
        train_data = sample_data[:train_size]
        val_data = sample_data[train_size:]
        
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data)
        })
        
        self.dataset = dataset_dict
        logger.info(f"示例数据集创建完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")
        
        return dataset_dict
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """数据预处理函数"""
        inputs = []
        targets = []
        
        for content, summary in zip(examples['content'], examples['summary']):
            # 格式化输入
            prompt = format_prompt(content)
            inputs.append(prompt)
            targets.append(summary)
        
        # Tokenization
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors=None
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors=None
        )
        
        # 将labels添加到model_inputs中
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def prepare_dataset(self, use_cache: bool = True) -> DatasetDict:
        """准备训练数据集"""
        if self.dataset is None:
            # 首先尝试从ModelScope加载
            self.dataset = self.load_from_modelscope()
        
        # 应用预处理
        logger.info("正在预处理数据集...")
        processed_dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset['train'].column_names,
            load_from_cache_file=use_cache,
            desc="预处理数据集"
        )
        
        logger.info("数据集预处理完成")
        return processed_dataset
    
    def get_dataloader(
        self, 
        split: str = "train", 
        batch_size: int = 4, 
        shuffle: bool = True
    ) -> DataLoader:
        """获取数据加载器"""
        if self.dataset is None:
            self.prepare_dataset()
        
        return DataLoader(
            self.dataset[split],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator
        )
    
    def data_collator(self, features: List[Dict]) -> Dict:
        """数据整理函数"""
        import torch
        
        # 获取最大长度
        max_length = max([len(f['input_ids']) for f in features])
        
        batch = {}
        for key in ['input_ids', 'attention_mask', 'labels']:
            batch[key] = []
            
        for feature in features:
            # padding
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            labels = feature['labels']
            
            # pad to max_length
            pad_length = max_length - len(input_ids)
            
            input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
            labels.extend([-100] * pad_length)  # -100 will be ignored in loss calculation
            
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)
        
        # 转换为tensor
        for key in batch:
            batch[key] = torch.tensor(batch[key])
            
        return batch
    
    def save_dataset(self, output_path: str):
        """保存数据集"""
        if self.dataset is None:
            logger.warning("数据集未加载，无法保存")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.dataset.save_to_disk(output_path)
        logger.info(f"数据集已保存到: {output_path}")
    
    def get_sample_data(self, num_samples: int = 5) -> List[Dict]:
        """获取样本数据用于测试"""
        if self.dataset is None:
            self.load_from_modelscope()
        
        samples = []
        for i in range(min(num_samples, len(self.dataset['train']))):
            sample = self.dataset['train'][i]
            samples.append({
                'content': sample['content'],
                'summary': sample['summary']
            })
        
        return samples