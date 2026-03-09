#!/usr/bin/env python3
"""
测试DeepSeek嵌入API
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_deepseek_embedding():
    """测试DeepSeek嵌入API"""
    print("🧪 测试DeepSeek嵌入API...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    
    if not api_key:
        print("❌ 未找到OPENAI_API_KEY")
        return False
    
    url = f"{base_url}/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 尝试不同的嵌入模型
    models_to_try = [
        "text-embedding-3-small",
        "text-embedding-ada-002", 
        "deepseek-embedding",
        "embedding"
    ]
    
    for model in models_to_try:
        print(f"\n尝试模型: {model}")
        
        payload = {
            "model": model,
            "input": "This is a test sentence for embedding."
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ DeepSeek嵌入API工作正常 (模型: {model})")
                embeddings = result.get('data', [{}])[0].get('embedding', [])
                print(f"嵌入向量维度: {len(embeddings)}")
                return True, model
            else:
                print(f"❌ 模型 {model} 失败: {response.text}")
                
        except Exception as e:
            print(f"❌ 模型 {model} 异常: {e}")
    
    return False, None

def main():
    print("🚀 开始测试DeepSeek嵌入API...")
    print("=" * 50)
    
    success, working_model = test_deepseek_embedding()
    
    print("\n" + "=" * 50)
    if success:
        print(f"🎉 找到可用的嵌入模型: {working_model}")
    else:
        print("⚠️ DeepSeek不支持嵌入API，需要使用其他方案")
        
    return success

if __name__ == "__main__":
    main()