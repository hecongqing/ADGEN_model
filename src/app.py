"""
Gradio可视化界面
"""
import os
import gradio as gr
import requests
import json
from typing import List, Dict, Optional
from loguru import logger

from .utils import setup_logger, format_prompt
from .model import QwenAdvertiseGenerator


class AdvertiseApp:
    """广告文案生成应用"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        service_url: Optional[str] = None,
        use_local_model: bool = True
    ):
        self.model_path = model_path
        self.service_url = service_url or "http://localhost:8000"
        self.use_local_model = use_local_model
        self.generator = None
        
        # 设置日志
        setup_logger("app.log", level="INFO")
        
        # 初始化模型（如果使用本地模型）
        if self.use_local_model:
            self._init_local_model()
    
    def _init_local_model(self):
        """初始化本地模型"""
        try:
            logger.info("正在初始化本地模型...")
            self.generator = QwenAdvertiseGenerator()
            
            if self.model_path and os.path.exists(self.model_path):
                self.generator.load_finetuned_model(self.model_path)
                logger.info(f"已加载微调模型: {self.model_path}")
            else:
                self.generator.load_model()
                logger.info("已加载预训练模型")
                
        except Exception as e:
            logger.error(f"本地模型初始化失败: {e}")
            self.generator = None
    
    def generate_local(
        self, 
        content: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        """使用本地模型生成"""
        if self.generator is None:
            return "本地模型未初始化，请检查模型路径或使用服务模式"
        
        try:
            result = self.generator.generate_text(
                content,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return result
        except Exception as e:
            logger.error(f"本地生成失败: {e}")
            return f"生成失败: {str(e)}"
    
    def generate_service(
        self, 
        content: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        """使用服务生成"""
        try:
            payload = {
                "prompt": content,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty
            }
            
            response = requests.post(
                f"{self.service_url}/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("generated_text", "服务返回错误")
            else:
                return f"服务请求失败: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"服务请求失败: {e}")
            return f"服务连接失败: {str(e)}"
    
    def generate_advertise(
        self,
        content: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        use_service: bool = False
    ) -> str:
        """生成广告文案"""
        if not content.strip():
            return "请输入商品特征信息"
        
        # 根据模式选择生成方式
        if use_service:
            return self.generate_service(
                content, max_new_tokens, temperature, top_p, repetition_penalty
            )
        else:
            return self.generate_local(
                content, max_new_tokens, temperature, top_p, repetition_penalty
            )
    
    def batch_generate(
        self,
        content_list: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_service: bool = False
    ) -> str:
        """批量生成广告文案"""
        if not content_list.strip():
            return "请输入商品特征信息列表"
        
        # 解析输入
        contents = [line.strip() for line in content_list.split('\n') if line.strip()]
        
        if not contents:
            return "请输入有效的商品特征信息"
        
        results = []
        for i, content in enumerate(contents, 1):
            try:
                result = self.generate_advertise(
                    content, max_new_tokens, temperature, top_p, 1.1, use_service
                )
                results.append(f"{i}. 输入: {content}\n   输出: {result}\n")
            except Exception as e:
                results.append(f"{i}. 输入: {content}\n   输出: 生成失败 - {str(e)}\n")
        
        return "\n".join(results)
    
    def get_examples(self) -> List[List[str]]:
        """获取示例数据"""
        examples = [
            [
                "类型#裙*版型#显瘦*材质#网纱*风格#性感*图案#蕾丝*图案#镂空*图案#纯色*裙下摆#鱼尾*裙长#连衣裙",
                256, 0.7, 0.9, 1.1, False
            ],
            [
                "类型#裤*版型#宽松*材质#牛仔*颜色#蓝色*风格#休闲*款式#直筒",
                256, 0.7, 0.9, 1.1, False
            ],
            [
                "类型#上衣*版型#修身*材质#棉*颜色#白色*风格#简约*领型#圆领*袖长#长袖",
                256, 0.7, 0.9, 1.1, False
            ],
            [
                "类型#鞋*材质#皮革*颜色#黑色*风格#商务*鞋跟#平底*款式#正装",
                256, 0.7, 0.9, 1.1, False
            ],
            [
                "类型#包*材质#帆布*颜色#米色*风格#文艺*容量#大容量*款式#单肩包",
                256, 0.7, 0.9, 1.1, False
            ]
        ]
        return examples
    
    def create_interface(self) -> gr.Interface:
        """创建Gradio界面"""
        
        with gr.Blocks(title="Qwen广告文案生成器", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🚀 Qwen广告文案生成器
                
                基于Qwen3模型的智能广告文案生成工具，支持根据商品特征自动生成吸引人的广告文案。
                
                ## 📝 使用说明
                1. 在输入框中输入商品特征，格式如：`类型#裙*版型#显瘦*材质#网纱*风格#性感`
                2. 调整生成参数（可选）
                3. 点击"生成广告文案"按钮
                4. 可以选择使用本地模型或远程服务
                """
            )
            
            with gr.Tab("单个生成"):
                with gr.Row():
                    with gr.Column(scale=2):
                        content_input = gr.Textbox(
                            label="商品特征",
                            placeholder="请输入商品特征，如：类型#裙*版型#显瘦*材质#网纱*风格#性感*图案#蕾丝",
                            lines=3,
                            max_lines=5
                        )
                        
                        with gr.Row():
                            max_tokens = gr.Slider(
                                minimum=50,
                                maximum=512,
                                value=256,
                                step=1,
                                label="最大生成长度"
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="创意度 (Temperature)"
                            )
                        
                        with gr.Row():
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="多样性 (Top-p)"
                            )
                            repetition_penalty = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=1.1,
                                step=0.05,
                                label="重复惩罚"
                            )
                        
                        with gr.Row():
                            use_service = gr.Checkbox(
                                label="使用远程服务",
                                value=False,
                                info="勾选后将使用远程vLLM服务，否则使用本地模型"
                            )
                        
                        generate_btn = gr.Button("🎯 生成广告文案", variant="primary")
                    
                    with gr.Column(scale=2):
                        output_text = gr.Textbox(
                            label="生成的广告文案",
                            lines=6,
                            max_lines=10,
                            interactive=False
                        )
                
                # 示例
                gr.Examples(
                    examples=self.get_examples(),
                    inputs=[content_input, max_tokens, temperature, top_p, repetition_penalty, use_service],
                    outputs=output_text,
                    fn=self.generate_advertise,
                    cache_examples=False
                )
                
                generate_btn.click(
                    fn=self.generate_advertise,
                    inputs=[content_input, max_tokens, temperature, top_p, repetition_penalty, use_service],
                    outputs=output_text
                )
            
            with gr.Tab("批量生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input = gr.Textbox(
                            label="商品特征列表",
                            placeholder="每行输入一个商品特征，例如：\n类型#裙*版型#显瘦*材质#网纱\n类型#裤*版型#宽松*材质#牛仔",
                            lines=8,
                            max_lines=15
                        )
                        
                        with gr.Row():
                            batch_max_tokens = gr.Slider(
                                minimum=50,
                                maximum=512,
                                value=256,
                                step=1,
                                label="最大生成长度"
                            )
                            batch_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="创意度"
                            )
                        
                        with gr.Row():
                            batch_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="多样性"
                            )
                            batch_use_service = gr.Checkbox(
                                label="使用远程服务",
                                value=False
                            )
                        
                        batch_generate_btn = gr.Button("🚀 批量生成", variant="primary")
                    
                    with gr.Column(scale=2):
                        batch_output = gr.Textbox(
                            label="批量生成结果",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
                
                batch_generate_btn.click(
                    fn=self.batch_generate,
                    inputs=[batch_input, batch_max_tokens, batch_temperature, batch_top_p, batch_use_service],
                    outputs=batch_output
                )
            
            with gr.Tab("📊 模型信息"):
                gr.Markdown(
                    f"""
                    ## 模型配置
                    
                    - **模型路径**: {self.model_path or "使用预训练模型"}
                    - **服务地址**: {self.service_url}
                    - **本地模型状态**: {"✅ 已加载" if self.generator else "❌ 未加载"}
                    
                    ## 功能特点
                    
                    - 🎯 基于Qwen3大语言模型
                    - 🚀 支持LoRA微调优化
                    - 📊 可调节生成参数
                    - 🔄 支持批量生成
                    - 🌐 支持本地模型和远程服务
                    
                    ## 输入格式说明
                    
                    商品特征格式：`属性#值*属性#值*...`
                    
                    常用属性：
                    - 类型: 裙、裤、上衣、鞋、包等
                    - 版型: 显瘦、宽松、修身、直筒等
                    - 材质: 网纱、牛仔、棉、皮革等
                    - 风格: 性感、休闲、简约、商务等
                    - 颜色: 蓝色、白色、黑色等
                    """
                )
        
        return interface
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        **kwargs
    ):
        """启动应用"""
        interface = self.create_interface()
        
        logger.info(f"启动Gradio应用: http://{server_name}:{server_port}")
        
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            **kwargs
        )


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="广告文案生成应用")
    parser.add_argument("--model_path", type=str, default=None,
                        help="微调模型路径")
    parser.add_argument("--service_url", type=str, default="http://localhost:8000",
                        help="远程服务地址")
    parser.add_argument("--use_local_model", action="store_true", default=True,
                        help="是否使用本地模型")
    parser.add_argument("--server_name", type=str, default="0.0.0.0",
                        help="服务器地址")
    parser.add_argument("--server_port", type=int, default=7860,
                        help="服务器端口")
    parser.add_argument("--share", action="store_true",
                        help="是否创建公共链接")
    
    args = parser.parse_args()
    
    # 创建应用
    app = AdvertiseApp(
        model_path=args.model_path,
        service_url=args.service_url,
        use_local_model=args.use_local_model
    )
    
    # 启动应用
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )


if __name__ == "__main__":
    main()