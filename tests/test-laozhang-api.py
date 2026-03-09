#!/usr/bin/env python3
"""
测试老张API的兼容性
"""

import requests
import json

def test_laozhang_chat_completions():
    """测试老张API的Chat Completions端点"""
    print("🧪 测试老张API Chat Completions端点...")
    
    api_key = "sk-rmMS3NM1iiJI7BkzF153946dCaA4491a9cD73907F7001834"
    url = "https://api.laozhang.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
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
            print("✅ Chat Completions端点工作正常")
            content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"响应: {content}")
            print(f"模型: {result.get('model', 'unknown')}")
            return True
        else:
            print(f"❌ Chat Completions端点失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat Completions端点异常: {e}")
        return False

def test_laozhang_responses():
    """测试老张API的Responses端点"""
    print("\n🧪 测试老张API Responses端点...")
    
    api_key = "sk-rmMS3NM1iiJI7BkzF153946dCaA4491a9cD73907F7001834"
    url = "https://api.laozhang.ai/v1/responses"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "input": "Hello, this is a test of the Responses API.",
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Responses端点工作正常！")
            print(f"响应结构: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ Responses端点失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Responses端点异常: {e}")
        return False

def test_laozhang_embeddings():
    """测试老张API的Embeddings端点"""
    print("\n🧪 测试老张API Embeddings端点...")
    
    api_key = "sk-rmMS3NM1iiJI7BkzF153946dCaA4491a9cD73907F7001834"
    url = "https://api.laozhang.ai/v1/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "text-embedding-ada-002",
        "input": "This is a test embedding."
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Embeddings端点工作正常")
            
            embeddings = result.get('data', [])
            if embeddings:
                embedding = embeddings[0].get('embedding', [])
                print(f"嵌入维度: {len(embedding)}")
                print(f"前5个值: {embedding[:5]}")
            
            return True
        else:
            print(f"❌ Embeddings端点失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Embeddings端点异常: {e}")
        return False

def main():
    print("🚀 测试老张API兼容性...")
    print("=" * 60)
    
    results = []
    
    # 测试各个端点
    results.append(("Chat Completions", test_laozhang_chat_completions()))
    results.append(("Responses API", test_laozhang_responses()))
    results.append(("Embeddings", test_laozhang_embeddings()))
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📋 测试结果:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 支持" if passed else "❌ 不支持"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if results[1][1]:  # Responses API支持
        print("🎉 老张API支持Responses API！")
        print("💡 可以直接使用，不需要API代理")
        print("🔧 建议：更新配置直接连接老张API")
    elif results[0][1]:  # 只支持Chat Completions
        print("⚠️  老张API只支持Chat Completions")
        print("💡 需要继续使用API代理进行转换")
        print("🔧 建议：更新代理目标为老张API")
    else:
        print("❌ 老张API连接失败")
        print("💡 请检查API密钥和网络连接")
    
    return results

if __name__ == "__main__":
    main()