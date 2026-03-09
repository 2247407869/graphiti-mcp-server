#!/usr/bin/env python3
"""
测试API代理功能
"""

import requests
import json
import time

def test_proxy_health():
    """测试代理健康状态"""
    print("🧪 测试API代理健康状态...")
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ API代理健康检查通过")
            print(f"   服务: {data.get('service')}")
            print(f"   目标: {data.get('target')}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_responses_api():
    """测试Responses API代理"""
    print("\n🧪 测试Responses API代理...")
    
    url = "http://localhost:8001/v1/responses"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "input": "Hello, this is a test of the Responses API proxy.",
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Responses API代理工作正常")
            
            # 显示响应结构
            print(f"   响应ID: {data.get('id')}")
            print(f"   模型: {data.get('model')}")
            print(f"   输出项数量: {len(data.get('output', []))}")
            
            # 显示第一个输出项的内容
            output = data.get('output', [])
            if output:
                first_item = output[0]
                print(f"   内容: {first_item.get('content', '')[:100]}...")
            
            return True
        else:
            print(f"❌ Responses API代理失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Responses API代理异常: {e}")
        return False

def test_chat_completions_passthrough():
    """测试Chat Completions直通"""
    print("\n🧪 测试Chat Completions直通...")
    
    url = "http://localhost:8001/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "Hello, this is a test of the Chat Completions passthrough."}
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat Completions直通工作正常")
            
            # 显示响应结构
            print(f"   响应ID: {data.get('id')}")
            print(f"   模型: {data.get('model')}")
            
            choices = data.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                print(f"   内容: {message.get('content', '')[:100]}...")
            
            return True
        else:
            print(f"❌ Chat Completions直通失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat Completions直通异常: {e}")
        return False

def test_embeddings_passthrough():
    """测试Embeddings直通"""
    print("\n🧪 测试Embeddings直通...")
    
    url = "http://localhost:8001/v1/embeddings"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": "This is a test embedding."
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Embeddings直通工作正常")
            
            # 显示响应结构
            embeddings = data.get('data', [])
            if embeddings:
                embedding = embeddings[0].get('embedding', [])
                print(f"   嵌入维度: {len(embedding)}")
                print(f"   前5个值: {embedding[:5]}")
            
            return True
        else:
            print(f"❌ Embeddings直通失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Embeddings直通异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 测试API代理功能...")
    print("=" * 60)
    
    # 等待服务启动
    print("⏳ 等待API代理启动...")
    time.sleep(5)
    
    results = []
    
    # 测试健康状态
    results.append(("健康检查", test_proxy_health()))
    
    # 测试各个端点
    results.append(("Responses API代理", test_responses_api()))
    results.append(("Chat Completions直通", test_chat_completions_passthrough()))
    results.append(("Embeddings直通", test_embeddings_passthrough()))
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📋 测试结果:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！API代理工作正常")
        print("💡 现在Graphiti应该可以通过代理正常工作了")
    else:
        print("⚠️  部分测试失败，请检查配置")
    
    return all_passed

if __name__ == "__main__":
    main()