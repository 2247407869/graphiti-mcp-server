#!/usr/bin/env python3
"""
直接测试SiliconFlow的正确端点
"""

import requests
import json

def test_correct_endpoint():
    """测试正确的聊天端点"""
    print("🧪 测试SiliconFlow正确的聊天端点...")
    
    api_key = "sk-gyowdkndmteuykdkamicbqdpcczdlmurlfdrcduyonoqtzwo"
    
    # 正确的端点
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
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
            print("✅ 正确端点工作正常")
            content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"响应: {content}")
            return True
        else:
            print(f"❌ 正确端点失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 正确端点异常: {e}")
        return False

def test_wrong_endpoint():
    """测试错误的端点（Graphiti使用的）"""
    print("\n🧪 测试错误的端点（Graphiti使用的）...")
    
    api_key = "sk-gyowdkndmteuykdkamicbqdpcczdlmurlfdrcduyonoqtzwo"
    
    # 错误的端点（Graphiti使用的）
    url = "https://api.siliconflow.cn/v1/responses"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
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
            print("✅ 错误端点居然工作了？")
            content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"响应: {content}")
            return True
        else:
            print(f"❌ 错误端点失败（预期的）: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 错误端点异常: {e}")
        return False

def main():
    print("🚀 验证SiliconFlow端点问题...")
    print("=" * 60)
    
    correct_ok = test_correct_endpoint()
    wrong_ok = test_wrong_endpoint()
    
    print("\n" + "=" * 60)
    print("📋 结论:")
    print(f"✅ 正确端点 (/chat/completions): {'工作' if correct_ok else '失败'}")
    print(f"❌ 错误端点 (/responses): {'工作' if wrong_ok else '失败（预期）'}")
    
    if correct_ok and not wrong_ok:
        print("\n🎯 确认：问题是Graphiti使用了错误的端点")
        print("💡 解决方案：需要修复Graphiti的端点配置")
    
    return correct_ok

if __name__ == "__main__":
    main()