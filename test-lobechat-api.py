#!/usr/bin/env python3
"""
测试 LobeChat API 配置
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_lobechat_api():
    """测试 LobeChat API 配置"""
    print("🔍 测试 LobeChat API 配置...")
    
    # 测试 LobeChat 健康状态
    try:
        response = requests.get("http://localhost:3210/api/health", timeout=5)
        print(f"LobeChat 健康检查: {response.status_code}")
        if response.status_code == 200:
            print("✅ LobeChat 运行正常")
        else:
            print(f"❌ LobeChat 健康检查失败: {response.text}")
    except Exception as e:
        print(f"❌ LobeChat 连接失败: {e}")
    
    # 测试 API 配置端点
    try:
        response = requests.get("http://localhost:3210/api/config", timeout=5)
        print(f"API 配置状态: {response.status_code}")
        if response.status_code == 200:
            config = response.json()
            print("✅ API 配置获取成功")
            print(f"   - OpenAI 配置: {config.get('openai', {})}")
        else:
            print(f"❌ API 配置获取失败: {response.text}")
    except Exception as e:
        print(f"❌ API 配置检查失败: {e}")
    
    # 测试 MCP 配置
    try:
        response = requests.get("http://localhost:3210/api/mcp", timeout=5)
        print(f"MCP 配置状态: {response.status_code}")
        if response.status_code == 200:
            mcp_config = response.json()
            print("✅ MCP 配置获取成功")
            print(f"   - MCP 服务器: {mcp_config}")
        else:
            print(f"❌ MCP 配置获取失败: {response.text}")
    except Exception as e:
        print(f"❌ MCP 配置检查失败: {e}")

def test_direct_api_call():
    """直接测试 API 调用"""
    print("\n🔍 测试直接 API 调用...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        print(f"直接 API 调用状态: {response.status_code}")
        if response.status_code == 200:
            print("✅ 直接 API 调用成功")
            result = response.json()
            print(f"   - 响应: {result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')}")
        else:
            print(f"❌ 直接 API 调用失败: {response.text}")
    except Exception as e:
        print(f"❌ 直接 API 调用异常: {e}")

if __name__ == "__main__":
    test_lobechat_api()
    test_direct_api_call()