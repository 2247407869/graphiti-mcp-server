#!/usr/bin/env python3
"""
OpenAI Responses API到Chat Completions API的代理转换器

解决Graphiti使用新的/v1/responses端点，但SiliconFlow只支持/v1/chat/completions的兼容性问题
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import aiohttp
from aiohttp import web
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("APIProxy")

class ResponsesToCompletionsProxy:
    """将OpenAI Responses API调用转换为Chat Completions API调用的代理"""
    
    def __init__(self):
        self.target_base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        self.api_key = os.getenv("OPENAI_API_KEY")
        
    def convert_responses_to_completions(self, responses_payload: Dict[str, Any]) -> Dict[str, Any]:
        """将Responses API请求转换为Chat Completions API请求"""
        
        # 提取基本参数
        completions_payload = {
            "model": responses_payload.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            "messages": [],
            "max_tokens": responses_payload.get("max_tokens", 1000),
            "temperature": responses_payload.get("temperature", 0.7),
            "stream": responses_payload.get("stream", False)
        }
        
        # 处理输入格式
        if "input" in responses_payload:
            input_data = responses_payload["input"]
            
            # 如果input是字符串，转换为user消息
            if isinstance(input_data, str):
                completions_payload["messages"] = [
                    {"role": "user", "content": input_data}
                ]
            # 如果input是消息列表，直接使用
            elif isinstance(input_data, list):
                completions_payload["messages"] = input_data
            else:
                # 尝试从其他字段提取消息
                completions_payload["messages"] = [
                    {"role": "user", "content": str(input_data)}
                ]
        
        # 处理instructions（系统提示）
        if "instructions" in responses_payload:
            instructions = responses_payload["instructions"]
            # 在消息列表开头添加系统消息
            completions_payload["messages"].insert(0, {
                "role": "system", 
                "content": instructions
            })
        
        # 处理messages字段（如果存在）
        if "messages" in responses_payload:
            completions_payload["messages"] = responses_payload["messages"]
        
        # 如果没有消息，创建默认消息
        if not completions_payload["messages"]:
            completions_payload["messages"] = [
                {"role": "user", "content": "Hello"}
            ]
        
        # 复制其他兼容的参数
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if key in responses_payload:
                completions_payload[key] = responses_payload[key]
        
        logger.info(f"转换请求: {len(completions_payload['messages'])} 条消息")
        return completions_payload
    
    def convert_completions_to_responses(self, completions_response: Dict[str, Any]) -> Dict[str, Any]:
        """将Chat Completions API响应转换为Responses API响应"""
        
        # 基本响应结构
        responses_response = {
            "id": completions_response.get("id", "resp_proxy_generated"),
            "object": "response",
            "created_at": completions_response.get("created", 0),
            "model": completions_response.get("model", "unknown"),
            "output": []
        }
        
        # 转换choices为output items
        choices = completions_response.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            
            # 创建消息item - 确保content是列表格式
            content = message.get("content", "")
            if isinstance(content, str):
                # 将字符串内容转换为content数组格式
                content_array = [{"type": "text", "text": content}] if content else []
            else:
                content_array = content if isinstance(content, list) else []
            
            message_item = {
                "id": f"msg_proxy_{choice.get('index', 0)}",
                "type": "message",
                "role": message.get("role", "assistant"),
                "content": content_array,
                "refusal": message.get("refusal", None),
                "finish_reason": choice.get("finish_reason", "stop")
            }
            
            responses_response["output"].append(message_item)
        
        # 复制usage信息
        if "usage" in completions_response:
            responses_response["usage"] = completions_response["usage"]
        
        logger.info(f"转换响应: {len(responses_response['output'])} 个输出项")
        return responses_response
    
    async def proxy_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """代理请求到目标API"""
        
        # 转换请求
        completions_request = self.convert_responses_to_completions(request_data)
        
        # 发送到目标API
        target_url = f"{self.target_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    target_url, 
                    json=completions_request, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        completions_response = await response.json()
                        # 转换响应
                        responses_response = self.convert_completions_to_responses(completions_response)
                        return responses_response
                    else:
                        error_text = await response.text()
                        logger.error(f"目标API错误 {response.status}: {error_text}")
                        raise Exception(f"API错误: {response.status} - {error_text}")
                        
            except Exception as e:
                logger.error(f"代理请求失败: {e}")
                raise

# 创建代理实例
proxy = ResponsesToCompletionsProxy()

# Web服务器
async def handle_responses(request):
    """处理/v1/responses端点"""
    try:
        # 解析请求
        request_data = await request.json()
        logger.info(f"收到Responses API请求: {request_data.get('model', 'unknown')}")
        
        # 代理请求
        response_data = await proxy.proxy_request(request_data)
        
        return web.json_response(response_data)
        
    except Exception as e:
        logger.error(f"处理请求失败: {e}")
        return web.json_response(
            {"error": {"message": str(e), "type": "proxy_error"}}, 
            status=500
        )

async def handle_chat_completions(request):
    """直接转发/v1/chat/completions端点"""
    try:
        # 解析请求
        request_data = await request.json()
        logger.info(f"收到Chat Completions请求: {request_data.get('model', 'unknown')}")
        
        # 直接转发到目标API
        target_url = f"{proxy.target_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {proxy.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                target_url, 
                json=request_data, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                response_data = await response.json()
                return web.json_response(response_data, status=response.status)
                
    except Exception as e:
        logger.error(f"转发请求失败: {e}")
        return web.json_response(
            {"error": {"message": str(e), "type": "proxy_error"}}, 
            status=500
        )

async def handle_embeddings(request):
    """直接转发/v1/embeddings端点"""
    try:
        # 解析请求
        request_data = await request.json()
        logger.info(f"收到Embeddings请求: {request_data.get('model', 'unknown')}")
        
        # 直接转发到目标API
        target_url = f"{proxy.target_base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {proxy.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                target_url, 
                json=request_data, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                response_data = await response.json()
                return web.json_response(response_data, status=response.status)
                
    except Exception as e:
        logger.error(f"转发Embeddings请求失败: {e}")
        return web.json_response(
            {"error": {"message": str(e), "type": "proxy_error"}}, 
            status=500
        )

async def handle_health(request):
    """健康检查"""
    return web.json_response({
        "status": "healthy",
        "service": "OpenAI API Proxy",
        "target": proxy.target_base_url,
        "endpoints": {
            "responses": "/v1/responses -> /v1/chat/completions",
            "chat_completions": "/v1/chat/completions -> /v1/chat/completions", 
            "embeddings": "/v1/embeddings -> /v1/embeddings"
        }
    })

def create_app():
    """创建Web应用"""
    app = web.Application()
    
    # 添加路由
    app.router.add_post('/v1/responses', handle_responses)
    app.router.add_post('/v1/chat/completions', handle_chat_completions)
    app.router.add_post('/v1/embeddings', handle_embeddings)
    app.router.add_get('/health', handle_health)
    app.router.add_get('/', handle_health)
    
    return app

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI API代理服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8001, help="服务器端口")
    
    args = parser.parse_args()
    
    if not proxy.api_key:
        logger.error("❌ 未找到OPENAI_API_KEY环境变量")
        return
    
    logger.info(f"🚀 启动OpenAI API代理服务器 {args.host}:{args.port}")
    logger.info(f"🎯 目标API: {proxy.target_base_url}")
    
    app = create_app()
    web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()