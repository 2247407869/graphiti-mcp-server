#!/usr/bin/env python3
"""
Graphiti MCP Server - 老张API版本

基于官方Graphiti框架的MCP服务器实现，使用老张API
支持Episode管理、搜索和基本的知识图谱操作
老张API原生支持OpenAI Responses API，无需代理转换
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# 尝试导入Graphiti和OpenAI客户端
try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    GRAPHITI_AVAILABLE = True
    LAOZHANG_AVAILABLE = True
except ImportError as e:
    GRAPHITI_AVAILABLE = False
    LAOZHANG_AVAILABLE = False
    print(f"⚠️  Graphiti或老张API客户端未安装: {e}")
    print("将使用模拟模式")

# 加载环境变量
load_dotenv()

# 配置日志
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GraphitiMCP")

# 配置
class Config:
    # 老张API配置（用于LLM推理和嵌入）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.laozhang.ai/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    SMALL_MODEL_NAME = os.getenv("SMALL_MODEL_NAME", os.getenv("MODEL_NAME", "gpt-4o-mini"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # 数据库配置
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    
    # Graphiti配置
    GRAPHITI_GROUP_ID = os.getenv("GRAPHITI_GROUP_ID", "demo")
    SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", "5"))
    
    # 服务器配置
    HOST = "0.0.0.0"
    PORT = 8000

# MCP请求/响应模型
class MCPRequest(BaseModel):
    method: str
    params: Dict[str, Any]

class MCPResponse(BaseModel):
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]

# Graphiti包装器
class GraphitiWrapper:
    def __init__(self):
        self.graphiti = None
        self.driver = None  # 显式存储驱动
        self.episodes = []  # 模拟存储
        
    async def initialize(self):
        """初始化Graphiti连接"""
        if not GRAPHITI_AVAILABLE or not LAOZHANG_AVAILABLE:
            logger.warning("Graphiti或老张API不可用，使用模拟模式")
            return True
            
        if not Config.OPENAI_API_KEY:
            logger.error("❌ 未找到OPENAI_API_KEY环境变量")
            return False
            
        try:
            # 使用FalkorDriver连接FalkorDB
            falkor_driver = FalkorDriver(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                username="default" if Config.REDIS_PASSWORD else None,
                password=Config.REDIS_PASSWORD if Config.REDIS_PASSWORD else None,
                database=Config.GRAPHITI_GROUP_ID  # 关键：指定正确的图名称
            )
            
            logger.info(f"🚀 初始化模型: LLM={Config.MODEL_NAME}, Embedding={Config.EMBEDDING_MODEL}")
            logger.info(f"🔗 API Endpoint: {Config.OPENAI_BASE_URL}")
            
            # 使用 OpenAIClient（/responses 端点）测试老张 API 兼容性
            self.graphiti = Graphiti(
                graph_driver=falkor_driver,
                llm_client=OpenAIClient(
                    config=LLMConfig(
                        api_key=Config.OPENAI_API_KEY,
                        model=Config.MODEL_NAME,
                        small_model=Config.SMALL_MODEL_NAME,
                        base_url=Config.OPENAI_BASE_URL
                    )
                ),
                embedder=OpenAIEmbedder(
                    config=OpenAIEmbedderConfig(
                        api_key=Config.OPENAI_API_KEY,
                        embedding_model=Config.EMBEDDING_MODEL,
                        base_url=Config.OPENAI_BASE_URL
                    )
                )
            )
            self.driver = falkor_driver  # 存储引用
            
            # gpt-4o-mini 原生兼容 Graphiti，无需额外补丁

            # 构建索引和约束
            await self.graphiti.build_indices_and_constraints()
            logger.info("✅ Graphiti初始化成功 (gpt-4o-mini)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Graphiti初始化失败: {e}")
            return False
    
    async def add_episode(self, name: str, episode_body: str, episode_type: str = "text", 
                          source_description: str = "User input", reference_time: Optional[str] = None) -> Dict[str, Any]:
        """添加Episode到知识图谱"""
        if not self.graphiti:
            return {
                "success": False,
                "message": "Graphiti未初始化，使用模拟模式",
                "episode_id": f"sim_{datetime.now().timestamp()}",
                "name": name,
                "content": episode_body
            }
            
        try:
            # 解析参考时间
            ref_time = None
            if reference_time:
                try:
                    ref_time = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))
                except ValueError:
                    ref_time = datetime.now(timezone.utc)
            else:
                ref_time = datetime.now(timezone.utc)
            
            # 确定Episode类型
            ep_type = EpisodeType.text
            if episode_type.lower() == "message":
                ep_type = EpisodeType.message
            elif episode_type.lower() == "observation":
                ep_type = EpisodeType.observation
            
            # 添加Episode
            result = await self.graphiti.add_episode(
                name=name,
                episode_body=episode_body,
                source=ep_type,  # 使用 source 参数而不是 episode_type
                source_description=source_description,
                reference_time=ref_time,
                group_id=Config.GRAPHITI_GROUP_ID
            )
            
            logger.info(f"✅ 成功添加Episode: {name}")
            
            # 将AddEpisodeResults对象转换为可序列化的字典
            episode_uuid = None
            if hasattr(result, 'episode') and result.episode:
                episode_uuid = str(result.episode.uuid) if hasattr(result.episode, 'uuid') else None
            
            return {
                "success": True,
                "message": "Episode添加成功",
                "episode_id": episode_uuid,
                "name": name,
                "content": episode_body,
                "episode_type": episode_type,
                "source_description": source_description,
                "reference_time": ref_time.isoformat() if ref_time else None,
                "group_id": Config.GRAPHITI_GROUP_ID,
                "created_nodes": len(result.nodes) if hasattr(result, 'nodes') else 0,
                "created_edges": len(result.edges) if hasattr(result, 'edges') else 0
            }
            
        except Exception as e:
            logger.error(f"❌ 添加Episode失败: {e}")
            return {
                "success": False,
                "message": f"添加Episode失败: {str(e)}",
                "error": str(e)
            }
    
    async def search_episodes(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """搜索记忆并返回格式化结果"""
        logger.info(f"🔎 搜索查询: '{query}', 限制: {num_results}")
        if not self.graphiti:
            return []

        try:
            results = await self.graphiti.search(
                query=query,
                num_results=num_results,
                group_ids=[Config.GRAPHITI_GROUP_ID]
            )
            
            formatted_results = []
            for res in results:
                # 基础字段对齐
                item = {
                    "id": str(getattr(res, 'uuid', 'unknown')),
                    "name": getattr(res, 'name', 'Unnamed'),
                    "score": float(getattr(res, 'score', 1.0)),
                    "created_at": datetime.now().isoformat(),
                    "source_description": getattr(res, 'source_description', 'Memory')
                }
                
                # 时间格式化
                if hasattr(res, 'created_at') and hasattr(res.created_at, 'isoformat'):
                    item["created_at"] = res.created_at.isoformat()
                
                # 类型识别逻辑
                if hasattr(res, 'fact'):  # 结构化知识 (Edge)
                    item["content"] = f"知识关系({item['name']})：{getattr(res, 'fact', '')}"
                    item["episode_type"] = "knowledge"
                    item["content_type"] = "知识关系"
                else:  # 原始记忆 (EpisodicNode)
                    item["content"] = getattr(res, 'content', '')
                    item["episode_type"] = "episode"
                    item["content_type"] = "原始记忆"
                
                formatted_results.append(item)
            
            # 按分数排序
            formatted_results.sort(key=lambda x: -x['score'])
            logger.info(f"✅ 搜索完成，找到 {len(formatted_results)} 条结果")
            return formatted_results
                
        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return []
    
    async def get_episodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """按时间顺序获取记忆列表"""
        if not self.graphiti:
            return []

        try:
            nodes = await self.graphiti.retrieve_episodes(
                reference_time=datetime.now(timezone.utc),
                last_n=limit,
                group_ids=[Config.GRAPHITI_GROUP_ID]
            )
            
            formatted_results = []
            for node in nodes:
                item = {
                    "id": str(getattr(node, 'uuid', 'unknown')),
                    "name": getattr(node, 'name', 'Unnamed'),
                    "content": getattr(node, 'content', ''),
                    "score": 1.0,
                    "created_at": datetime.now().isoformat(),
                    "episode_type": "episode",
                    "source_description": getattr(node, 'source_description', 'Original Memory'),
                    "content_type": "原始记忆"
                }
                
                if hasattr(node, 'created_at') and hasattr(node.created_at, 'isoformat'):
                    item["created_at"] = node.created_at.isoformat()
                formatted_results.append(item)
            
            logger.info(f"✅ 获取到 {len(formatted_results)} 条记忆")
            return formatted_results
                
        except Exception as e:
            logger.error(f"❌ 获取记忆列表失败: {e}")
            return []
    
    async def close(self):
        """关闭连接"""
        if self.graphiti:
            await self.graphiti.close()

# 全局Graphiti实例
graphiti_wrapper = GraphitiWrapper()

# FastAPI应用
app = FastAPI(
    title="Graphiti MCP Server",
    description="Graphiti知识图谱的MCP协议服务器（老张API版本）",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    logger.info("🚀 启动Graphiti MCP服务器（老张API版本）...")
    await graphiti_wrapper.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理"""
    logger.info("🛑 关闭Graphiti MCP服务器...")
    await graphiti_wrapper.close()

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "Graphiti MCP Server (老张API)",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "graphiti_available": GRAPHITI_AVAILABLE,
        "laozhang_available": LAOZHANG_AVAILABLE,
        "mode": "real" if graphiti_wrapper.graphiti else "simulation",
        "laozhang_configured": bool(Config.OPENAI_API_KEY),
        "direct_connection": True,
        "api_provider": "老张API"
    }

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Graphiti MCP Server (老张API)",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "tools": "/tools/list",
            "call": "/tools/call"
        }
    }

@app.get("/tools/list")
async def list_tools():
    """列出可用工具"""
    tools = [
        {
            "name": "add_episode",
            "description": "向知识图谱添加一个Episode（记忆片段）",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Episode名称"},
                    "episode_body": {"type": "string", "description": "Episode内容"},
                    "episode_type": {"type": "string", "description": "Episode类型", "default": "text"},
                    "source_description": {"type": "string", "description": "来源描述", "default": "MCP"}
                },
                "required": ["name", "episode_body"]
            }
        },
        {
            "name": "search",
            "description": "搜索知识图谱中的Episodes",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "num_results": {"type": "integer", "description": "返回结果数量", "default": 5}
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_episodes",
            "description": "获取Episodes列表",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回数量限制", "default": 100}
                }
            }
        }
    ]
    
    return {"tools": tools}

@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """调用工具"""
    try:
        tool_name = request.name
        args = request.arguments
        
        if tool_name == "add_episode":
            result = await graphiti_wrapper.add_episode(
                name=args.get("name", "Unnamed Episode"),
                episode_body=args.get("episode_body", ""),
                episode_type=args.get("episode_type", "text"),
                source_description=args.get("source_description", "MCP")
            )
            return {"result": result}
            
        elif tool_name == "search":
            results = await graphiti_wrapper.search_episodes(
                query=args.get("query", ""),
                num_results=args.get("num_results", 5)
            )
            return {"result": results}
            
        elif tool_name == "get_episodes":
            episodes = await graphiti_wrapper.get_episodes(
                limit=args.get("limit", 100)
            )
            return {"result": episodes}
            
        else:
            raise HTTPException(status_code=400, detail=f"未知工具: {tool_name}")
            
    except Exception as e:
        logger.error(f"工具调用失败: {e}")
        return {"error": {"code": -1, "message": str(e)}}

# MCP协议兼容端点
@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    """MCP协议端点"""
    try:
        if request.method == "tools/list":
            tools_response = await list_tools()
            return MCPResponse(result=tools_response["tools"])
            
        elif request.method == "tools/call":
            tool_request = ToolCallRequest(
                name=request.params.get("name"),
                arguments=request.params.get("arguments", {})
            )
            call_response = await call_tool(tool_request)
            return MCPResponse(result=call_response.get("result"), error=call_response.get("error"))
            
        else:
            return MCPResponse(error={"code": -1, "message": f"未知方法: {request.method}"})
            
    except Exception as e:
        logger.error(f"MCP请求处理失败: {e}")
        return MCPResponse(error={"code": -1, "message": str(e)})

@app.api_route("/mcp/stream", methods=["GET", "POST"])
async def mcp_stream_endpoint(request: Request):
    """MCP流式协议端点 - 兼容LobeChat"""
    # 处理GET请求 - 返回Manifest
    if request.method == "GET":
        logger.info("📥 MCP Stream Probe (GET) - Returning Manifest")
        return await mcp_manifest_get()

    try:
        # 解析请求体
        body = await request.json()
        logger.info(f"📥 MCP Stream Request: {json.dumps(body, ensure_ascii=False)}")
        
        # 处理不同的MCP方法
        if body.get("method") == "initialize":
            # 初始化响应
            return {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "graphiti-memory",
                        "version": "1.0.0",
                        "description": "Graphiti知识图谱记忆管理服务（老张API版本）"
                    }
                }
            }
        
        elif body.get("method") == "tools/list":
            # 工具列表
            tools_response = await list_tools()
            return {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "tools": tools_response["tools"]
                }
            }
        
        elif body.get("method") == "tools/call":
            # 工具调用
            params = body.get("params", {})
            tool_request = ToolCallRequest(
                name=params.get("name"),
                arguments=params.get("arguments", {})
            )
            call_response = await call_tool(tool_request)
            
            # MCP工具调用响应格式 - 直接返回工具结果
            if "result" in call_response:
                # 将结果转换为字符串格式，便于LobeChat显示
                result_data = call_response["result"]
                if isinstance(result_data, (dict, list)):
                    result_text = json.dumps(result_data, ensure_ascii=False, indent=2)
                else:
                    result_text = str(result_data)
                
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result_text
                            }
                        ]
                    }
                }
            elif "error" in call_response:
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": call_response["error"]
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32603,
                        "message": "Internal error: Invalid response format"
                    }
                }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {body.get('method')}"
                }
            }
            
    except Exception as e:
        logger.error(f"MCP流式请求处理失败: {e}")
        return {
            "jsonrpc": "2.0",
            "id": body.get("id") if 'body' in locals() else None,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }

@app.get("/mcp/capabilities")
async def mcp_capabilities():
    """MCP能力查询端点"""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "graphiti-memory",
            "version": "1.0.0",
            "description": "Graphiti知识图谱记忆管理服务（DeepSeek版本）"
        }
    }

@app.get("/mcp/manifest")
async def mcp_manifest_get():
    """MCP Manifest端点 - GET请求（直接返回manifest）"""
    tools = await list_tools()
    return {
        "name": "graphiti-memory",
        "version": "1.0.0",
        "description": "Graphiti知识图谱记忆管理服务（gpt-4o-mini）",
        "author": "Graphiti Demo",
        "homepage": "http://localhost:3000",
        "repository": "https://github.com/getzep/graphiti",
        "capabilities": {
            "tools": {}
        },
        "tools": tools["tools"],
        "serverInfo": {
            "name": "graphiti-memory",
            "version": "1.0.0"
        }
    }

@app.post("/mcp/manifest")
async def mcp_manifest_post(request: Request):
    """MCP Manifest端点 - POST请求（支持多种格式）"""
    try:
        # 尝试解析请求体
        try:
            body = await request.json()
        except:
            body = {}
        
        tools = await list_tools()
        manifest = {
            "name": "graphiti-memory",
            "version": "1.0.0",
            "description": "Graphiti知识图谱记忆管理服务（gpt-4o-mini）",
            "author": "Graphiti Demo",
            "homepage": "http://localhost:3000",
            "repository": "https://github.com/getzep/graphiti",
            "capabilities": {
                "tools": {}
            },
            "tools": tools["tools"],
            "serverInfo": {
                "name": "graphiti-memory",
                "version": "1.0.0"
            }
        }
        
        # 如果是JSON-RPC请求，返回JSON-RPC格式
        if isinstance(body, dict) and "jsonrpc" in body:
            return {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": manifest
            }
        else:
            # 否则直接返回manifest
            return manifest
            
    except Exception as e:
        logger.error(f"Manifest请求失败: {e}")
        # 尝试返回JSON-RPC错误格式
        try:
            body = await request.json()
            if isinstance(body, dict) and "jsonrpc" in body:
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -1,
                        "message": str(e)
                    }
                }
        except:
            pass
        
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Graphiti MCP Server (老张API)")
    parser.add_argument("--host", default=Config.HOST, help="服务器主机")
    parser.add_argument("--port", type=int, default=Config.PORT, help="服务器端口")
    parser.add_argument("--transport", default="sse", help="传输协议")
    
    args = parser.parse_args()
    
    logger.info(f"启动Graphiti MCP服务器（老张API版本） {args.host}:{args.port}")
    
    uvicorn.run(
        "graphiti_mcp_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()