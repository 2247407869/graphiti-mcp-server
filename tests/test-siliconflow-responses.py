#!/usr/bin/env python3
"""
直接测试SiliconFlow是否支持/v1/responses端点
"""

import requests
import json

def test_siliconflow_responses_api():
    """测试SiliconFlow的/v1/responses端点"""
    print("🧪 测试SiliconFlow是否支持/v1/responses端点...")
    
    api_key = "sk-gyowdkndmteuykdkamicbqdpcczdlmurlfdrcduyonoqtzwo"
    
    # 测试/v1/responses端点
    url = "https://api.siliconflow.cn/v1/responses"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 使用OpenAI Responses API格式
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "input": "Hello, this is a test of the Responses API.",
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SiliconFlow支持/v1/responses端点！")
            print(f"响应结构: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ SiliconFlow不支持/v1/responses端点")
            print(f"错误响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_different_responses_formats():
    """测试不同的Responses API格式"""
    print("\n🧪 测试不同的Responses API请求格式...")
    
    api_key = "sk-gyowdkndmteuykdkamicbqdpcczdlmurlfdrcduyonoqtzwo"
    url = "https://api.siliconflow.cn/v1/responses"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 测试格式1：简单input
    formats = [
        {
            "name": "简单input格式",
            "payload": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "input": "Hello world",
                "max_tokens": 20
            }
        },
        {
            "name": "messages格式",
            "payload": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [
                    {"role": "user", "content": "Hello world"}
                ],
                "max_tokens": 20
            }
        },
        {
            "name": "带instructions格式",
            "payload": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "input": "Hello world",
                "instructions": "You are a helpful assistant.",
                "max_tokens": 20
            }
        }
    ]
    
    for format_test in formats:
        print(f"\n   测试 {format_test['name']}...")
        try:
            response = requests.post(url, headers=headers, json=format_test['payload'], timeout=30)
            print(f"   状态码: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ {format_test['name']} 成功")
            else:
                print(f"   ❌ {format_test['name']} 失败: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ {format_test['name']} 异常: {e}")

def main():
    print("🚀 测试SiliconFlow对Responses API的支持...")
    print("=" * 60)
    
    # 基本测试
    basic_support = test_siliconflow_responses_api()
    
    if basic_support:
        # 如果基本支持，测试不同格式
        test_different_responses_formats()
        
        print("\n" + "=" * 60)
        print("🎉 SiliconFlow支持Responses API！")
        print("💡 这意味着我们不需要API代理，可以直接使用")
    else:
        print("\n" + "=" * 60)
        print("❌ SiliconFlow不支持Responses API")
        print("💡 需要使用API代理进行转换")
    
    return basic_support

if __name__ == "__main__":
    main()