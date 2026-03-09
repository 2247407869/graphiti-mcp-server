#!/usr/bin/env python3
"""
测试MCP集成功能
验证Graphiti MCP服务器的各项功能
"""

import requests
import json
import time

# 配置
MCP_SERVER_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    print("🔍 测试健康检查...")
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过: {data['status']}")
            print(f"   模式: {data['mode']}")
            print(f"   Graphiti可用: {data['graphiti_available']}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_manifest():
    """测试Manifest端点"""
    print("\n🔍 测试Manifest端点...")
    try:
        response = requests.get(f"{MCP_SERVER_URL}/mcp/manifest")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Manifest获取成功")
            print(f"   插件名称: {data['name']}")
            print(f"   版本: {data['version']}")
            print(f"   工具数量: {len(data['tools'])}")
            for tool in data['tools']:
                print(f"   - {tool['name']}: {tool['description']}")
            return True
        else:
            print(f"❌ Manifest获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Manifest获取异常: {e}")
        return False

def test_add_episode():
    """测试添加Episode"""
    print("\n🔍 测试添加Episode...")
    try:
        payload = {
            "name": "add_episode",
            "arguments": {
                "name": "测试记忆",
                "episode_body": "这是一个测试记忆片段，用于验证MCP集成功能。",
                "episode_type": "text",
                "source_description": "MCP测试"
            }
        }
        
        response = requests.post(f"{MCP_SERVER_URL}/tools/call", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get("result", {}).get("success"):
                print(f"✅ Episode添加成功")
                print(f"   Episode ID: {data['result']['episode_id']}")
                print(f"   消息: {data['result']['message']}")
                return data['result']['episode_id']
            else:
                print(f"❌ Episode添加失败: {data.get('result', {}).get('error', '未知错误')}")
                return None
        else:
            print(f"❌ Episode添加请求失败: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Episode添加异常: {e}")
        return None

def test_search():
    """测试搜索功能"""
    print("\n🔍 测试搜索功能...")
    try:
        payload = {
            "name": "search",
            "arguments": {
                "query": "测试",
                "num_results": 5
            }
        }
        
        response = requests.post(f"{MCP_SERVER_URL}/tools/call", json=payload)
        if response.status_code == 200:
            data = response.json()
            results = data.get("result", [])
            print(f"✅ 搜索完成，找到 {len(results)} 个结果")
            for i, result in enumerate(results):
                print(f"   {i+1}. {result.get('name', 'Unnamed')}")
                print(f"      内容: {result.get('content', '')[:50]}...")
                print(f"      相似度: {result.get('score', 0):.2f}")
            return len(results) > 0
        else:
            print(f"❌ 搜索请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 搜索异常: {e}")
        return False

def test_get_episodes():
    """测试获取Episodes列表"""
    print("\n🔍 测试获取Episodes列表...")
    try:
        payload = {
            "name": "get_episodes",
            "arguments": {
                "limit": 10
            }
        }
        
        response = requests.post(f"{MCP_SERVER_URL}/tools/call", json=payload)
        if response.status_code == 200:
            data = response.json()
            episodes = data.get("result", [])
            print(f"✅ 获取Episodes成功，共 {len(episodes)} 个")
            for i, episode in enumerate(episodes):
                print(f"   {i+1}. {episode.get('name', 'Unnamed')}")
                print(f"      ID: {episode.get('id', 'Unknown')}")
                print(f"      创建时间: {episode.get('created_at', 'Unknown')}")
            return True
        else:
            print(f"❌ 获取Episodes失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取Episodes异常: {e}")
        return False

def test_mcp_protocol():
    """测试MCP协议端点"""
    print("\n🔍 测试MCP协议端点...")
    try:
        # 测试工具列表
        payload = {
            "method": "tools/list",
            "params": {}
        }
        
        response = requests.post(f"{MCP_SERVER_URL}/mcp/stream", json=payload)
        if response.status_code == 200:
            data = response.json()
            tools = data.get("result", [])
            print(f"✅ MCP协议工具列表获取成功，共 {len(tools)} 个工具")
            return True
        else:
            print(f"❌ MCP协议测试失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MCP协议测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始MCP集成功能测试\n")
    
    tests = [
        ("健康检查", test_health),
        ("Manifest端点", test_manifest),
        ("添加Episode", test_add_episode),
        ("搜索功能", test_search),
        ("获取Episodes", test_get_episodes),
        ("MCP协议", test_mcp_protocol)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
        
        time.sleep(1)  # 短暂延迟
    
    print(f"\n{'='*50}")
    print(f"测试总结: {passed}/{total} 通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有测试通过！MCP集成功能正常工作。")
        print("\n📋 下一步:")
        print("1. 访问 http://localhost:3210 打开LobeChat")
        print("2. 配置DeepSeek API密钥")
        print("3. 在插件设置中添加MCP插件:")
        print("   - 插件标识符: graphiti-memory")
        print("   - Manifest URL: http://graphiti-mcp:8000/mcp/manifest")
        print("4. 开始与具有记忆功能的AI助手对话！")
    else:
        print("⚠️  部分测试失败，请检查服务状态。")
        print("\n🔧 故障排除:")
        print("1. 检查Docker容器状态: docker-compose ps")
        print("2. 查看MCP服务器日志: docker-compose logs graphiti-mcp")
        print("3. 检查网络连接和端口占用")

if __name__ == "__main__":
    main()