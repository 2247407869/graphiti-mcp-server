#!/usr/bin/env python3
"""
测试 LobeChat 配置状态
"""

import requests
import json
import time

def test_lobechat_ready():
    """测试 LobeChat 是否准备就绪"""
    print("🔍 检查 LobeChat 状态...")
    
    try:
        # 检查主页
        response = requests.get("http://localhost:3210", timeout=5)
        if response.status_code == 200:
            print("✅ LobeChat 主页可访问")
        else:
            print(f"❌ LobeChat 主页访问失败: {response.status_code}")
            return False
            
        # 检查聊天页面
        response = requests.get("http://localhost:3210/chat", timeout=5)
        if response.status_code == 200:
            print("✅ LobeChat 聊天页面可访问")
        else:
            print(f"❌ LobeChat 聊天页面访问失败: {response.status_code}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ LobeChat 连接失败: {e}")
        return False

def check_mcp_integration():
    """检查 MCP 集成状态"""
    print("\n🔍 检查 MCP 集成状态...")
    
    try:
        # 检查 MCP 服务器
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ MCP 服务器运行正常")
            print(f"   - 模式: {health.get('mode', 'unknown')}")
            print(f"   - Graphiti 可用: {health.get('graphiti_available', False)}")
            print(f"   - API 提供商: {health.get('api_provider', 'unknown')}")
        else:
            print(f"❌ MCP 服务器健康检查失败: {response.status_code}")
            return False
            
        # 检查 MCP 工具
        response = requests.get("http://localhost:8000/tools/list", timeout=5)
        if response.status_code == 200:
            tools = response.json()
            print(f"✅ MCP 工具可用: {len(tools.get('tools', []))} 个")
            for tool in tools.get('tools', []):
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print(f"❌ MCP 工具列表获取失败: {response.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ MCP 服务器连接失败: {e}")
        return False

def print_setup_instructions():
    """打印设置说明"""
    print("\n" + "="*60)
    print("🎯 LobeChat 设置说明")
    print("="*60)
    print()
    print("1. 打开浏览器访问: http://localhost:3210")
    print()
    print("2. 如果遇到 'Failed to fetch' 错误，请在 LobeChat 设置中配置:")
    print("   📝 API 密钥: sk-rmMS3NM1iiJI7BkzF153946dCaA4491a9cD73907F7001834")
    print("   🌐 API 端点: https://api.laozhang.ai/v1")
    print("   🤖 模型: gpt-3.5-turbo")
    print()
    print("3. 启用 MCP 插件（记忆功能）:")
    print("   🔌 插件名称: graphiti-memory")
    print("   🌐 服务器URL: http://graphiti-mcp:8000/mcp/stream")
    print("   📡 传输方式: http")
    print()
    print("4. 开始对话，AI 将具有长期记忆功能！")
    print()
    print("📖 详细设置指南: 查看 LOBECHAT_SETUP_GUIDE.md")
    print("="*60)

def main():
    """主函数"""
    print("🚀 LobeChat 配置检查")
    print("="*40)
    
    # 等待服务启动
    print("⏳ 等待服务启动...")
    time.sleep(2)
    
    # 检查 LobeChat
    lobechat_ok = test_lobechat_ready()
    
    # 检查 MCP
    mcp_ok = check_mcp_integration()
    
    # 总结
    print("\n" + "="*40)
    print("📊 检查结果:")
    print(f"   LobeChat: {'✅ 正常' if lobechat_ok else '❌ 异常'}")
    print(f"   MCP 服务: {'✅ 正常' if mcp_ok else '❌ 异常'}")
    
    if lobechat_ok and mcp_ok:
        print("\n🎉 所有服务运行正常！")
        print_setup_instructions()
    else:
        print("\n⚠️  部分服务异常，请检查 Docker 容器状态")
        print("   运行: docker-compose ps")
        print("   查看日志: docker-compose logs")

if __name__ == "__main__":
    main()