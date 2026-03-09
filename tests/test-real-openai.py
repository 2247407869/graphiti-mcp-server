#!/usr/bin/env python3
"""
测试真正的OpenAI API作为对比
"""

import requests
import json
import os

def test_real_openai_responses():
    """测试真正的OpenAI Responses API"""
    print("🧪 测试真正的OpenAI Responses API...")
    
    # 注意：这需要真正的OpenAI API密钥
    # 这里只是为了演示格式，不会真正调用
    print("⚠️  这需要真正的OpenAI API密钥，跳过实际调用")
    
    # 预期的响应格式（基于文档）
    expected_format = {
        "id": "resp_abc123",
        "object": "response",
        "created_at": 1234567890,
        "model": "gpt-4o",
        "output": [
            {
                "id": "msg_abc123",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello! How can I help you today?"
                    }
                ],
                "refusal": None
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }
    
    print("📋 预期的OpenAI Responses API格式:")
    print(json.dumps(expected_format, indent=2, ensure_ascii=False))
    
    return expected_format

def compare_with_proxy():
    """对比我们的代理响应"""
    print("\n🔍 对比我们的代理响应...")
    
    url = "http://localhost:8001/v1/responses"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "input": "Hello! How can I help you today?",
        "max_tokens": 20
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            proxy_response = response.json()
            print("📋 我们的代理响应格式:")
            print(json.dumps(proxy_response, indent=2, ensure_ascii=False))
            
            # 检查关键字段
            print("\n🔍 关键字段检查:")
            print(f"   object: {proxy_response.get('object')}")
            print(f"   output类型: {type(proxy_response.get('output'))}")
            
            if proxy_response.get('output'):
                first_output = proxy_response['output'][0]
                print(f"   第一个输出类型: {first_output.get('type')}")
                print(f"   content类型: {type(first_output.get('content'))}")
                print(f"   refusal存在: {'refusal' in first_output}")
                print(f"   refusal值: {first_output.get('refusal')}")
            
            return proxy_response
        else:
            print(f"❌ 代理响应失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return None

def main():
    print("🚀 对比OpenAI Responses API格式...")
    print("=" * 60)
    
    # 显示预期格式
    expected = test_real_openai_responses()
    
    # 对比我们的代理
    proxy_response = compare_with_proxy()
    
    print("\n" + "=" * 60)
    print("📋 格式对比结论:")
    
    if proxy_response:
        # 检查关键差异
        differences = []
        
        if proxy_response.get('object') != 'response':
            differences.append("object字段不匹配")
        
        if not isinstance(proxy_response.get('output'), list):
            differences.append("output不是数组")
        elif proxy_response.get('output'):
            first_output = proxy_response['output'][0]
            if first_output.get('type') != 'message':
                differences.append("输出项type不是message")
            if not isinstance(first_output.get('content'), list):
                differences.append("content不是数组")
            if 'refusal' not in first_output:
                differences.append("缺少refusal字段")
        
        if differences:
            print("❌ 发现格式差异:")
            for diff in differences:
                print(f"   - {diff}")
        else:
            print("✅ 格式基本匹配OpenAI标准")
            print("💡 问题可能在Graphiti的解析逻辑中")
    else:
        print("❌ 无法获取代理响应进行对比")

if __name__ == "__main__":
    main()