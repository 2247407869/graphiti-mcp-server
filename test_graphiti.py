#!/usr/bin/env python3
"""
Graphiti演示测试脚本

基于官方示例测试Graphiti + FalkorDB的基本功能
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any

# 如果需要直接测试Graphiti（可选）
try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    print("⚠️  graphiti-core未安装，将使用MCP API测试")

import requests


class GraphitiTester:
    def __init__(self, mcp_url: str = "http://localhost:8000"):
        self.mcp_url = mcp_url
        self.graphiti = None
        
    async def init_direct_graphiti(self):
        """初始化直接Graphiti连接（如果可用）"""
        if not GRAPHITI_AVAILABLE:
            return False
            
        try:
            self.graphiti = Graphiti(
                uri="falkor://localhost:6379",
                # 如果需要认证
                # username="default",
                # password=""
            )
            await self.graphiti.build_indices_and_constraints()
            print("✅ 直接Graphiti连接成功")
            return True
        except Exception as e:
            print(f"❌ 直接Graphiti连接失败: {e}")
            return False
    
    def test_mcp_connection(self) -> bool:
        """测试MCP服务器连接"""
        try:
            response = requests.get(f"{self.mcp_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ MCP服务器连接成功")
                return True
            else:
                print(f"❌ MCP服务器响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ MCP服务器连接失败: {e}")
            return False
    
    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP工具"""
        try:
            response = requests.post(
                f"{self.mcp_url}/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_direct_episode_operations(self):
        """测试直接Graphiti Episode操作"""
        if not self.graphiti:
            print("⏭️  跳过直接Graphiti测试（未连接）")
            return
        
        print("\n🧪 测试直接Graphiti Episode操作...")
        
        try:
            # 添加文本Episode
            episode_body = """
            Alice和Bob在AI会议上讨论了Graphiti项目。
            他们决定使用FalkorDB作为后端存储，
            并计划在下周开始实施。
            """
            
            episode_id = await self.graphiti.add_episode(
                name="AI会议记录",
                episode_body=episode_body.strip(),
                episode_type=EpisodeType.text,
                reference_time=datetime.now(timezone.utc),
                source_description="测试脚本"
            )
            
            print(f"✅ Episode添加成功，ID: {episode_id}")
            
            # 搜索测试
            search_results = await self.graphiti.search(
                query="Alice和Bob讨论了什么？",
                num_results=5
            )
            
            print(f"✅ 搜索完成，找到 {len(search_results)} 个结果")
            for i, result in enumerate(search_results[:3]):
                print(f"   {i+1}. {result}")
                
        except Exception as e:
            print(f"❌ 直接Graphiti测试失败: {e}")
    
    def test_mcp_episode_operations(self):
        """测试MCP Episode操作"""
        print("\n🧪 测试MCP Episode操作...")
        
        # 测试添加Episode
        add_result = self.call_mcp_tool("add_episode", {
            "name": "MCP测试记录",
            "episode_body": "这是通过MCP协议添加的测试记忆。包含了项目进展和技术决策。",
            "episode_type": "text",
            "source_description": "MCP测试脚本"
        })
        
        if "error" in add_result:
            print(f"❌ MCP添加Episode失败: {add_result['error']}")
        else:
            print("✅ MCP添加Episode成功")
            print(f"   结果: {add_result}")
        
        # 测试搜索
        search_result = self.call_mcp_tool("search", {
            "query": "项目进展",
            "num_results": 5
        })
        
        if "error" in search_result:
            print(f"❌ MCP搜索失败: {search_result['error']}")
        else:
            print("✅ MCP搜索成功")
            results = search_result.get("result", [])
            print(f"   找到 {len(results)} 个结果")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. {result}")
    
    def test_batch_episodes(self):
        """测试批量Episode操作"""
        print("\n🧪 测试批量Episode操作...")
        
        episodes = [
            {
                "name": "技术会议1",
                "content": "讨论了微服务架构的优缺点，决定采用Docker容器化部署。",
            },
            {
                "name": "产品规划",
                "content": "确定了Q1季度的产品路线图，重点关注用户体验优化。",
            },
            {
                "name": "团队建设",
                "content": "新增了两名前端开发工程师，团队规模扩大到12人。",
            }
        ]
        
        success_count = 0
        for episode in episodes:
            result = self.call_mcp_tool("add_episode", {
                "name": episode["name"],
                "episode_body": episode["content"],
                "episode_type": "text",
                "source_description": "批量测试"
            })
            
            if "error" not in result:
                success_count += 1
            else:
                print(f"   ❌ {episode['name']} 添加失败: {result['error']}")
        
        print(f"✅ 批量添加完成: {success_count}/{len(episodes)} 成功")
        
        # 测试相关搜索
        test_queries = ["技术架构", "产品规划", "团队"]
        for query in test_queries:
            result = self.call_mcp_tool("search", {"query": query, "num_results": 3})
            if "error" not in result:
                results = result.get("result", [])
                print(f"   '{query}' 搜索: {len(results)} 个结果")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始Graphiti功能测试...\n")
        
        # 测试MCP连接
        if not self.test_mcp_connection():
            print("❌ MCP服务器不可用，请先启动服务")
            return
        
        # 测试直接Graphiti连接（可选）
        await self.init_direct_graphiti()
        
        # 运行各项测试
        await self.test_direct_episode_operations()
        self.test_mcp_episode_operations()
        self.test_batch_episodes()
        
        print("\n🎉 测试完成！")
        print("\n📱 现在可以访问Web界面进行交互测试:")
        print("   http://localhost:3000")
        
        if self.graphiti:
            await self.graphiti.close()


async def main():
    """主函数"""
    tester = GraphitiTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())