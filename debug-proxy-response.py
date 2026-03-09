#!/usr/bin/env python3
"""
调试API代理响应格式
"""

import requests
import json

def test_proxy_response_format():
    """测试代理响应格式"""
    print("🔍 调试API代理响应格式...")
    
    url = "http://localhost:8001/v1/responses"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "input": "Hello, this is a debug test.",
        "max_tokens": 20
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 代理响应成功")
            print("📋 完整响应结构:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # 检查output结构
            output = data.get("output", [])
            if output:
                first_item = output[0]
                print(f"\n🔍 第一个输出项详细结构:")
                print(f"   ID: {first_item.get('id')}")
                print(f"   类型: {first_item.get('type')}")
                print(f"   角色: {first_item.get('role')}")
                print(f"   内容类型: {type(first_item.get('content'))}")
                print(f"   内容值: {first_item.get('content')}")
                print(f"   结束原因: {first_item.get('finish_reason')}")
            
            return True
        else:
            print(f"❌ 代理响应失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_direct_siliconflow_format():
    """测试直接SiliconFlow响应格式作为对比"""
    print("\n🔍 对比：直接SiliconFlow响应格式...")
    
    api_key = "sk-gyowdkndmteuykdkamicbqdpcczdlmurlfdrcduyonoqtzwo"
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "Hello, this is a debug test."}
        ],
        "max_tokens": 20
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 直接SiliconFlow响应成功")
            print("📋 原始Chat Completions响应结构:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            return True
        else:
            print(f"❌ 直接SiliconFlow响应失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def main():
    print("🚀 调试API代理响应格式...")
    print("=" * 60)
    
    # 测试代理响应
    proxy_ok = test_proxy_response_format()
    
    # 测试直接响应作为对比
    direct_ok = test_direct_siliconflow_format()
    
    print("\n" + "=" * 60)
    print("📋 调试结果:")
    print(f"   代理响应: {'✅ 成功' if proxy_ok else '❌ 失败'}")
    print(f"   直接响应: {'✅ 成功' if direct_ok else '❌ 失败'}")
    
    if proxy_ok and direct_ok:
        print("\n💡 两种响应都成功，可以对比格式差异")
    elif proxy_ok:
        print("\n💡 代理响应成功，但直接响应失败")
    elif direct_ok:
        print("\n💡 直接响应成功，但代理响应失败")
    else:
        print("\n💡 两种响应都失败，需要检查网络连接")

if __name__ == "__main__":
    main()