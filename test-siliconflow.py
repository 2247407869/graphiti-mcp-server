#!/usr/bin/env python3
"""
测试SiliconFlow API
"""

import requests
import json

def test_siliconflow_chat():
    """测试SiliconFlow聊天API"""
    print("🧪 测试SiliconFlow聊天API...")
    
    api_key = "sk-gyowdkndmteuykdkamicbqdpcczdlmurlfdrcduyonoqtzwo"
    base_url = "https://api.siliconflow.cn/v1"
    
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "Hello, this is a test message."}
        ],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SiliconFlow聊天API工作正常")
            content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"响应: {content}")
            return True
        else:
            print(f"❌ SiliconFlow聊天API失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ SiliconFlow聊天API异常: {e}")
        return False

def test_siliconflow_embedding():
    """测试SiliconFlow嵌入API"""
    print("\n🧪 测试SiliconFlow嵌入API...")
    
    api_key = "sk-gyowdkndmteuykdkamicbqdpcczdlmurlfdrcduyonoqtzwo"
    base_url = "https://api.siliconflow.cn/v1"
    
    url = f"{base_url}/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 尝试不同的嵌入模型
    models_to_try = [
        "BAAI/bge-large-zh-v1.5",
        "BAAI/bge-m3",
        "Qwen/Qwen3-Embedding-8B",
        "text-embedding-3-small"
    ]
    
    for model in models_to_try:
        print(f"\n尝试模型: {model}")
        
        payload = {
            "model": model,
            "input": "This is a test sentence for embedding.",
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ SiliconFlow嵌入API工作正常 (模型: {model})")
                embeddings = result.get('data', [{}])[0].get('embedding', [])
                print(f"嵌入向量维度: {len(embeddings)}")
                return True, model
            else:
                print(f"❌ 模型 {model} 失败: {response.text}")
                
        except Exception as e:
            print(f"❌ 模型 {model} 异常: {e}")
    
    return False, None

def main():
    print("🚀 开始测试SiliconFlow API...")
    print("=" * 60)
    
    chat_ok = test_siliconflow_chat()
    embedding_ok, working_embedding_model = test_siliconflow_embedding()
    
    print("\n" + "=" * 60)
    if chat_ok and embedding_ok:
        print("🎉 SiliconFlow API测试全部通过！")
        print(f"推荐聊天模型: Qwen/Qwen2.5-7B-Instruct")
        print(f"推荐嵌入模型: {working_embedding_model}")
    elif chat_ok:
        print("⚠️ 聊天API正常，但嵌入API有问题")
    else:
        print("❌ API测试失败")
        
    return chat_ok and embedding_ok

if __name__ == "__main__":
    main()