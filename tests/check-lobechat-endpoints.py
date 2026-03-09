#!/usr/bin/env python3
"""
检查 LobeChat 的实际 API 端点
"""

import requests
import json

def check_lobechat_endpoints():
    """检查 LobeChat 的可用端点"""
    print("🔍 检查 LobeChat 端点...")
    
    base_url = "http://localhost:3210"
    
    # 常见的 API 端点
    endpoints = [
        "/",
        "/chat",
        "/api",
        "/api/chat",
        "/api/chat/completions",
        "/api/openai",
        "/api/openai/chat/completions",
        "/api/v1/chat/completions",
        "/trpc",
        "/api/trpc",
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code != 404:
                print(f"✅ {endpoint}: {response.status_code}")
                if response.status_code == 200 and 'json' in response.headers.get('content-type', ''):
                    try:
                        data = response.json()
                        print(f"   数据: {json.dumps(data, indent=2)[:200]}...")
                    except:
                        pass
            else:
                print(f"❌ {endpoint}: 404")
        except Exception as e:
            print(f"❌ {endpoint}: 错误 - {e}")

def test_chat_completion():
    """测试聊天完成端点"""
    print("\n🔍 测试聊天完成端点...")
    
    endpoints = [
        "/api/chat/completions",
        "/api/openai/chat/completions", 
        "/api/v1/chat/completions"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.post(
                f"http://localhost:3210{endpoint}",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                },
                timeout=10
            )
            print(f"POST {endpoint}: {response.status_code}")
            if response.status_code != 404:
                print(f"   响应: {response.text[:200]}...")
        except Exception as e:
            print(f"POST {endpoint}: 错误 - {e}")

if __name__ == "__main__":
    check_lobechat_endpoints()
    test_chat_completion()