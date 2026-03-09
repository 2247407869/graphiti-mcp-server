#!/usr/bin/env python3
"""
测试记忆功能是否正常工作
"""

import requests
import json
import time

def test_mcp_server():
    """测试 MCP 服务器功能"""
    print("🔍 测试 MCP 服务器记忆功能...")
    
    base_url = "http://localhost:8000"
    
    # 1. 测试健康检查
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ MCP 服务器运行正常")
            print(f"   - 模式: {health.get('mode', 'unknown')}")
            print(f"   - Graphiti 可用: {health.get('graphiti_available', False)}")
        else:
            print(f"❌ MCP 服务器健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MCP 服务器连接失败: {e}")
        return False
    
    # 2. 测试添加记忆
    print("\n📝 测试添加记忆...")
    try:
        memory_data = {
            "name": "用户信息",
            "arguments": {
                "name": "用户信息",
                "episode_body": "用户的名字是李林松，是一名软件工程师，主要做后端开发，熟悉 Python 和 Java。",
                "episode_type": "text",
                "source_description": "用户自我介绍"
            }
        }
        
        response = requests.post(
            f"{base_url}/tools/call",
            headers={"Content-Type": "application/json"},
            json=memory_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 记忆添加成功")
            print(f"   - Episode ID: {result.get('result', {}).get('episode_id', 'N/A')}")
            print(f"   - 消息: {result.get('result', {}).get('message', 'N/A')}")
        else:
            print(f"❌ 记忆添加失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 记忆添加异常: {e}")
        return False
    
    # 3. 测试搜索记忆
    print("\n🔍 测试搜索记忆...")
    try:
        search_data = {
            "name": "search",
            "arguments": {
                "query": "李林松",
                "num_results": 5
            }
        }
        
        response = requests.post(
            f"{base_url}/tools/call",
            headers={"Content-Type": "application/json"},
            json=search_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            memories = result.get('result', [])
            print(f"✅ 搜索完成，找到 {len(memories)} 个相关记忆")
            
            for i, memory in enumerate(memories[:3]):  # 只显示前3个
                print(f"   {i+1}. ID: {memory.get('id', 'N/A')}")
                print(f"      内容: {memory.get('content', 'N/A')[:100]}...")
                print(f"      相似度: {memory.get('score', 'N/A')}")
                
            return len(memories) > 0
        else:
            print(f"❌ 搜索失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 搜索异常: {e}")
        return False

def test_mcp_stream_endpoint():
    """测试 MCP 流式端点（LobeChat 使用的）"""
    print("\n🌊 测试 MCP 流式端点...")
    
    try:
        # 测试工具列表
        list_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        response = requests.post(
            "http://localhost:8000/mcp/stream",
            headers={"Content-Type": "application/json"},
            json=list_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            tools = result.get('result', {}).get('tools', [])
            print(f"✅ MCP 流式端点正常，工具数量: {len(tools)}")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print(f"❌ MCP 流式端点失败: {response.status_code}")
            return False
            
        # 测试添加记忆（通过流式端点）
        add_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "add_episode",
                "arguments": {
                    "name": "测试记忆",
                    "episode_body": "这是通过 MCP 流式端点添加的测试记忆。用户名字是李林松。",
                    "episode_type": "text",
                    "source_description": "MCP 流式测试"
                }
            }
        }
        
        response = requests.post(
            "http://localhost:8000/mcp/stream",
            headers={"Content-Type": "application/json"},
            json=add_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 通过流式端点添加记忆成功")
            return True
        else:
            print(f"❌ 通过流式端点添加记忆失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ MCP 流式端点测试异常: {e}")
        return False

def check_existing_memories():
    """检查现有记忆"""
    print("\n📚 检查现有记忆...")
    
    try:
        search_data = {
            "name": "search",
            "arguments": {
                "query": "李林松 软件工程师 Python Java",
                "num_results": 10
            }
        }
        
        response = requests.post(
            "http://localhost:8000/tools/call",
            headers={"Content-Type": "application/json"},
            json=search_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            memories = result.get('result', [])
            print(f"📊 数据库中共有 {len(memories)} 个相关记忆")
            
            if memories:
                print("\n🔍 相关记忆内容:")
                for i, memory in enumerate(memories):
                    print(f"\n{i+1}. 记忆 ID: {memory.get('id', 'N/A')}")
                    print(f"   内容: {memory.get('content', 'N/A')}")
                    print(f"   相似度: {memory.get('score', 'N/A')}")
                    print(f"   创建时间: {memory.get('created_at', 'N/A')}")
            else:
                print("⚠️  没有找到相关记忆，可能需要重新添加")
                
            return len(memories)
        else:
            print(f"❌ 检查记忆失败: {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"❌ 检查记忆异常: {e}")
        return 0

def main():
    """主函数"""
    print("🧠 记忆功能诊断")
    print("="*50)
    
    # 测试 MCP 服务器基本功能
    mcp_ok = test_mcp_server()
    
    # 测试流式端点
    stream_ok = test_mcp_stream_endpoint()
    
    # 检查现有记忆
    memory_count = check_existing_memories()
    
    # 总结
    print("\n" + "="*50)
    print("📊 诊断结果:")
    print(f"   MCP 服务器: {'✅ 正常' if mcp_ok else '❌ 异常'}")
    print(f"   流式端点: {'✅ 正常' if stream_ok else '❌ 异常'}")
    print(f"   现有记忆: {memory_count} 个")
    
    if mcp_ok and stream_ok:
        print("\n🎉 记忆功能正常！")
        if memory_count == 0:
            print("💡 建议: 在 LobeChat 中重新介绍自己，让 AI 记住你的信息")
        print("\n📋 使用建议:")
        print("1. 在 LobeChat 中说: '我的名字是李林松，请记住这个信息'")
        print("2. 确保 MCP 插件已正确配置并启用")
        print("3. 在新对话中测试: '你还记得我的名字吗？'")
    else:
        print("\n⚠️  记忆功能异常，请检查:")
        print("1. MCP 服务器状态: docker-compose logs graphiti-mcp")
        print("2. LobeChat MCP 插件配置")
        print("3. 网络连接")

if __name__ == "__main__":
    main()