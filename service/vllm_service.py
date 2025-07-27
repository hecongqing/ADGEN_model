"""
基于vLLM的推理服务
"""
import os
import sys
import asyncio
from typing import Dict, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logger, format_prompt, clean_generated_text


# 请求和响应模型
class GenerateRequest(BaseModel):
    """生成请求模型"""
    prompt: str = Field(..., description="输入提示词")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="最大生成tokens数")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="采样温度")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="核采样参数")
    top_k: int = Field(50, ge=1, le=100, description="top-k采样参数")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="重复惩罚")
    do_sample: bool = Field(True, description="是否使用采样")
    stream: bool = Field(False, description="是否流式输出")


class GenerateResponse(BaseModel):
    """生成响应模型"""
    generated_text: str = Field(..., description="生成的文本")
    prompt: str = Field(..., description="输入提示词")
    finish_reason: str = Field(..., description="结束原因")
    usage: Dict = Field(..., description="使用统计")


class BatchGenerateRequest(BaseModel):
    """批量生成请求模型"""
    prompts: List[str] = Field(..., description="提示词列表")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="最大生成tokens数")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="采样温度")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="核采样参数")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="重复惩罚")


class VLLMService:
    """vLLM推理服务"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        dtype: str = "half",
        trust_remote_code: bool = True
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        
        self.llm = None
        self.sampling_params_default = None
        
    async def initialize(self):
        """初始化vLLM引擎"""
        try:
            from vllm import LLM, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            logger.info("正在初始化vLLM引擎...")
            
            # 创建vLLM引擎
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                enforce_eager=True,  # 避免CUDA图问题
            )
            
            # 默认采样参数
            self.sampling_params_default = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_tokens=256,
                repetition_penalty=1.1,
                use_beam_search=False,
                stop=None,
            )
            
            logger.info("vLLM引擎初始化完成")
            
        except Exception as e:
            logger.error(f"vLLM引擎初始化失败: {e}")
            raise
    
    def create_sampling_params(self, request: GenerateRequest):
        """创建采样参数"""
        from vllm import SamplingParams
        
        return SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty,
            use_beam_search=not request.do_sample,
            stop=None,
        )
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """生成文本"""
        if self.llm is None:
            raise HTTPException(status_code=500, detail="模型未初始化")
        
        try:
            # 格式化提示词
            formatted_prompt = format_prompt(request.prompt)
            
            # 创建采样参数
            sampling_params = self.create_sampling_params(request)
            
            # 生成
            outputs = self.llm.generate([formatted_prompt], sampling_params)
            
            if not outputs:
                raise HTTPException(status_code=500, detail="生成失败")
            
            output = outputs[0]
            
            # 提取生成的文本
            generated_text = output.outputs[0].text
            
            # 清理生成的文本
            generated_text = clean_generated_text(generated_text)
            
            # 计算使用统计
            usage = {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason
            }
            
            return GenerateResponse(
                generated_text=generated_text,
                prompt=request.prompt,
                finish_reason=output.outputs[0].finish_reason,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")
    
    async def batch_generate(self, request: BatchGenerateRequest) -> List[GenerateResponse]:
        """批量生成文本"""
        if self.llm is None:
            raise HTTPException(status_code=500, detail="模型未初始化")
        
        try:
            from vllm import SamplingParams
            
            # 格式化提示词
            formatted_prompts = [format_prompt(prompt) for prompt in request.prompts]
            
            # 创建采样参数
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_new_tokens,
                repetition_penalty=request.repetition_penalty,
                use_beam_search=False,
                stop=None,
            )
            
            # 批量生成
            outputs = self.llm.generate(formatted_prompts, sampling_params)
            
            results = []
            for i, output in enumerate(outputs):
                # 提取生成的文本
                generated_text = output.outputs[0].text
                generated_text = clean_generated_text(generated_text)
                
                # 计算使用统计
                usage = {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                    "finish_reason": output.outputs[0].finish_reason
                }
                
                results.append(GenerateResponse(
                    generated_text=generated_text,
                    prompt=request.prompts[i],
                    finish_reason=output.outputs[0].finish_reason,
                    usage=usage
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")


# 全局服务实例
vllm_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global vllm_service
    
    # 启动时初始化vLLM服务
    logger.info("正在启动vLLM服务...")
    
    model_path = os.getenv("MODEL_PATH", "./outputs")
    tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
    
    vllm_service = VLLMService(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    
    await vllm_service.initialize()
    logger.info("vLLM服务启动完成")
    
    yield
    
    # 关闭时清理资源
    logger.info("正在关闭vLLM服务...")


# 创建FastAPI应用
app = FastAPI(
    title="Qwen广告文案生成服务",
    description="基于vLLM的高性能Qwen模型推理服务",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """根路径"""
    return {"message": "Qwen广告文案生成服务正在运行"}


@app.get("/health")
async def health_check():
    """健康检查"""
    global vllm_service
    if vllm_service is None or vllm_service.llm is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return {"status": "healthy", "service": "vllm"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """生成文本接口"""
    global vllm_service
    if vllm_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    return await vllm_service.generate(request)


@app.post("/batch_generate", response_model=List[GenerateResponse])
async def batch_generate_text(request: BatchGenerateRequest):
    """批量生成文本接口"""
    global vllm_service
    if vllm_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    if len(request.prompts) > 100:
        raise HTTPException(status_code=400, detail="批量请求最多支持100个提示词")
    
    return await vllm_service.batch_generate(request)


@app.get("/info")
async def get_model_info():
    """获取模型信息"""
    global vllm_service
    if vllm_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    return {
        "model_path": vllm_service.model_path,
        "tensor_parallel_size": vllm_service.tensor_parallel_size,
        "gpu_memory_utilization": vllm_service.gpu_memory_utilization,
        "max_model_len": vllm_service.max_model_len,
        "dtype": vllm_service.dtype,
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM推理服务")
    parser.add_argument("--model_path", type=str, default="./outputs",
                        help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务器地址")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务器端口")
    parser.add_argument("--workers", type=int, default=1,
                        help="工作进程数")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU内存利用率")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="最大模型长度")
    parser.add_argument("--log_file", type=str, default="vllm_service.log",
                        help="日志文件")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)
    os.environ["GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
    os.environ["MAX_MODEL_LEN"] = str(args.max_model_len)
    
    # 设置日志
    setup_logger(args.log_file, level="INFO")
    
    # 启动服务
    logger.info(f"启动vLLM服务: http://{args.host}:{args.port}")
    
    uvicorn.run(
        "vllm_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
        access_log=True
    )


if __name__ == "__main__":
    main()