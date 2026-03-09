#!/usr/bin/env python3
"""
测试API密钥和端点
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_deepseek_api():
    """测试DeepSeek API"""
    print("🧪 测试DeepSeek API...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    
    if not api_key:
        print("❌ 未找到OPENAI_API_KEY")
        return False
    
    # 正确的端点应该是 /chat/completions
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Hello, this is a test."}
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ DeepSeek API工作正常")
            print(f"响应: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
            return True
        else:
            print(f"❌ DeepSeek API失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ DeepSeek API异常: {e}")
        return False

def test_dashscope_api():
    """测试阿里云DashScope API"""
    print("\n🧪 测试阿里云DashScope API...")
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    # 使用正确的端点
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    
    if not api_key:
        print("❌ 未找到DASHSCOPE_API_KEY")
        return False
    
    url = f"{base_url}/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "text-embedding-v4",
        "input": "This is a test sentence for embedding.",
        "dimensions": 1024,
        "encoding_format": "float"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 阿里云DashScope API工作正常")
            embeddings = result.get('data', [{}])[0].get('embedding', [])
            print(f"嵌入向量维度: {len(embeddings)}")
            return True
        else:
            print(f"❌ 阿里云DashScope API失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 阿里云DashScope API异常: {e}")
        return False

def main():
    print("🚀 开始测试API密钥和端点...")
    print("=" * 50)
    
    deepseek_ok = test_deepseek_api()
    dashscope_ok = test_dashscope_api()
    
    print("\n" + "=" * 50)
    if deepseek_ok and dashscope_ok:
        print("🎉 所有API测试通过！")
    else:
        print("⚠️ 部分API测试失败，需要检查配置")
        
    return deepseek_ok and dashscope_ok

if __name__ == "__main__":
    main()