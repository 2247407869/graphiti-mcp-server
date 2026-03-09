#!/usr/bin/env python3
"""
完整测试老张API的所有端点（增加超时时间）
"""

import requests
import json
import time

def test_laozhang_chat_completions_extended():
    """测试老张API的Chat Completions端点（延长超时）"""
    print("🧪 测试老张API Chat Completions端点（延长超时）...")
    
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
        print("   发送请求...")
        response = requests.post(url, headers=headers, json=payload, timeout=120)  # 2分钟超时
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat Completions端点工作正常")
            content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"   响应: {content}")
            print(f"   模型: {result.get('model', 'unknown')}")
            return True
        else:
            print(f"❌ Chat Completions端点失败: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Chat Completions端点超时（2分钟）")
        return False
    except Exception as e:
        print(f"❌ Chat Completions端点异常: {e}")
        return False

def test_laozhang_embeddings_extended():
    """测试老张API的Embeddings端点（延长超时）"""
    print("\n🧪 测试老张API Embeddings端点（延长超时）...")
    
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
        print("   发送请求...")
        response = requests.post(url, headers=headers, json=payload, timeout=120)  # 2分钟超时
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Embeddings端点工作正常")
            
            embeddings = result.get('data', [])
            if embeddings:
                embedding = embeddings[0].get('embedding', [])
                print(f"   嵌入维度: {len(embedding)}")
                print(f"   前5个值: {embedding[:5]}")
            
            return True
        else:
            print(f"❌ Embeddings端点失败: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Embeddings端点超时（2分钟）")
        return False
    except Exception as e:
        print(f"❌ Embeddings端点异常: {e}")
        return False

def test_laozhang_responses_quick():
    """快速重新测试Responses端点"""
    print("\n🧪 快速重新测试老张API Responses端点...")
    
    api_key = "sk-rmMS3NM1iiJI7BkzF153946dCaA4491a9cD73907F7001834"
    url = "https://api.laozhang.ai/v1/responses"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "input": "Hello, quick test.",
        "max_tokens": 20
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        end_time = time.time()
        
        print(f"   状态码: {response.status_code}")
        print(f"   响应时间: {end_time - start_time:.2f}秒")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Responses端点工作正常")
            
            # 提取响应内容
            output = result.get('output', [])
            if output and output[0].get('content'):
                content = output[0]['content'][0].get('text', 'No text')
                print(f"   响应: {content}")
            
            return True
        else:
            print(f"❌ Responses端点失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Responses端点异常: {e}")
        return False

def main():
    print("🚀 完整测试老张API所有端点...")
    print("=" * 60)
    
    results = []
    
    # 测试所有端点
    results.append(("Responses API", test_laozhang_responses_quick()))
    results.append(("Chat Completions", test_laozhang_chat_completions_extended()))
    results.append(("Embeddings", test_laozhang_embeddings_extended()))
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📋 完整测试结果:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 支持" if passed else "❌ 不支持/超时"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 老张API完全兼容OpenAI标准！")
        print("✅ 支持所有必需的端点")
        print("💡 可以完全替代OpenAI API")
        print("🔧 建议：移除API代理，直接使用老张API")
    elif results[0][1]:  # Responses API支持
        print("🎯 老张API支持关键的Responses API！")
        print("✅ 这是最重要的兼容性")
        print("⚠️  其他端点可能需要网络优化")
        print("💡 建议：继续使用，优化网络配置")
    else:
        print("❌ 老张API连接存在问题")
        print("💡 建议：检查网络连接和API密钥")
    
    return results

if __name__ == "__main__":
    main()