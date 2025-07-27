#!/usr/bin/env python3
"""
Qwen广告文案生成器使用示例
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import QwenAdvertiseGenerator
from src.dataset import AdvertiseGenDataset
from src.utils import setup_logger

def main():
    """主函数"""
    print("🚀 Qwen广告文案生成器示例")
    print("=" * 50)
    
    # 设置日志
    setup_logger("example.log", level="INFO")
    
    try:
        # 1. 加载数据集示例
        print("\n📊 1. 数据集示例")
        print("-" * 30)
        
        dataset_manager = AdvertiseGenDataset()
        samples = dataset_manager.get_sample_data(num_samples=3)
        
        for i, sample in enumerate(samples, 1):
            print(f"样本 {i}:")
            print(f"  输入: {sample['content']}")
            print(f"  输出: {sample['summary']}")
            print()
        
        # 2. 模型生成示例
        print("\n🤖 2. 模型生成示例")
        print("-" * 30)
        
        # 注意：这里会尝试加载预训练模型，需要较长时间和较大内存
        generator = QwenAdvertiseGenerator()
        
        print("正在加载模型...")
        try:
            generator.load_model()
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 提示: 请确保有足够的GPU内存，或尝试使用CPU模式")
            return
        
        # 测试生成
        test_inputs = [
            "类型#裙*版型#显瘦*材质#网纱*风格#性感*图案#蕾丝",
            "类型#裤*版型#宽松*材质#牛仔*颜色#蓝色*风格#休闲",
            "类型#上衣*版型#修身*材质#棉*颜色#白色*风格#简约"
        ]
        
        for i, content in enumerate(test_inputs, 1):
            print(f"测试 {i}:")
            print(f"  输入: {content}")
            
            try:
                result = generator.generate_text(
                    content,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9
                )
                print(f"  生成: {result}")
            except Exception as e:
                print(f"  生成失败: {e}")
            
            print()
        
        # 3. API调用示例
        print("\n🌐 3. API调用示例")
        print("-" * 30)
        
        print("如果你启动了vLLM服务，可以这样调用:")
        print("""
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "类型#裙*版型#显瘦*材质#网纱*风格#性感",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
})

result = response.json()
print(result["generated_text"])
        """)
        
        # 4. Web应用示例
        print("\n🎨 4. Web应用使用")
        print("-" * 30)
        
        print("启动Web应用:")
        print("  方法1: ./scripts/start_app.sh")
        print("  方法2: python -m src.app")
        print("  然后访问: http://localhost:7860")
        
    except KeyboardInterrupt:
        print("\n\n👋 示例已停止")
    except Exception as e:
        print(f"\n❌ 运行示例时出错: {e}")
        print("💡 请检查依赖是否正确安装")

if __name__ == "__main__":
    main()