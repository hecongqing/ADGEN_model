"""
模型定义和推理模块
"""
import os
import torch
from typing import Dict, List, Optional, Union, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from loguru import logger

from .utils import get_device, clean_generated_text, format_prompt


class QwenAdvertiseGenerator:
    """基于Qwen的广告文案生成器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        cache_dir: str = "./dataset",
        device: Optional[torch.device] = None,
        use_lora: bool = True,
        use_8bit: bool = True,
        max_memory: Optional[Dict] = None
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or get_device()
        self.use_lora = use_lora
        self.use_8bit = use_8bit
        self.max_memory = max_memory
        
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        self._setup_generation_config()
    
    def _setup_generation_config(self):
        """设置生成配置"""
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_beams=1,
            max_new_tokens=256,
            min_new_tokens=10,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=None,  # 将在加载tokenizer后设置
            eos_token_id=None,  # 将在加载tokenizer后设置
        )
    
    def load_model(self, model_path: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """加载模型和tokenizer"""
        model_path = model_path or self.model_name
        
        try:
            logger.info(f"正在加载tokenizer: {model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 更新生成配置
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
            logger.info(f"正在加载模型: {model_path}")
            
            # 配置模型加载参数
            model_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "trust_remote_code": True,
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            # 8bit量化配置
            if self.use_8bit and torch.cuda.is_available():
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            
            # 最大内存配置
            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            # 应用LoRA
            if self.use_lora:
                self.model = self._apply_lora(self.model)
            
            logger.info("模型加载完成")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """应用LoRA配置"""
        logger.info("正在应用LoRA配置...")
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        
        # 为量化训练准备模型
        if self.use_8bit:
            model = prepare_model_for_kbit_training(model)
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数数量
        model.print_trainable_parameters()
        
        return model
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """生成文本"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用load_model()方法")
        
        # 格式化输入
        formatted_prompt = format_prompt(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 移动到正确的设备
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 更新生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=True
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        
        # 清理生成的文本
        generated_text = clean_generated_text(generated_text)
        
        return generated_text
    
    def batch_generate(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 4
    ) -> List[str]:
        """批量生成文本"""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                try:
                    result = self.generate_text(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"生成失败: {e}")
                    batch_results.append("生成失败")
            
            results.extend(batch_results)
            logger.info(f"已完成 {len(results)}/{len(prompts)} 个样本的生成")
        
        return results
    
    def save_model(self, output_dir: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未加载，无法保存")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型和tokenizer
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            # 如果是PEFT模型，保存适配器
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存生成配置
        self.generation_config.save_pretrained(output_dir)
        
        logger.info(f"模型已保存到: {output_dir}")
    
    def load_finetuned_model(self, model_path: str):
        """加载微调后的模型"""
        try:
            logger.info(f"正在加载微调模型: {model_path}")
            
            # 先加载基础模型
            self.load_model()
            
            # 如果是LoRA模型，加载适配器
            if self.use_lora:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, model_path)
            else:
                # 加载完整模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            
            logger.info("微调模型加载完成")
            
        except Exception as e:
            logger.error(f"加载微调模型失败: {e}")
            raise


class QwenTrainer:
    """Qwen模型训练器"""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        output_dir: str = "./outputs",
        logging_dir: str = "./outputs/logs"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
    
    def get_training_arguments(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        dataloader_num_workers: int = 4,
        remove_unused_columns: bool = False,
        **kwargs
    ) -> TrainingArguments:
        """获取训练参数"""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            logging_dir=self.logging_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            fp16=fp16 and torch.cuda.is_available(),
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=remove_unused_columns,
            report_to="tensorboard",
            save_total_limit=3,
            **kwargs
        )
        
        return training_args
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        training_args: Optional[TrainingArguments] = None,
        data_collator=None
    ) -> Trainer:
        """训练模型"""
        
        if training_args is None:
            training_args = self.get_training_arguments()
        
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8 if training_args.fp16 else None
            )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        trainer.save_state()
        
        logger.info(f"训练完成，模型已保存到: {self.output_dir}")
        
        return trainer