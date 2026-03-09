#!/usr/bin/env python3
"""
测试Graphiti记忆功能
"""

import requests
import json
import time

def test_add_memory():
    """测试添加记忆"""
    url = "http://localhost:8000/tools/call"
    
    payload = {
        "name": "add_episode",
        "arguments": {
            "name": "测试记忆",
            "episode_body": "这是一个测试记忆片段，用于验证Graphiti配置是否正常工作。使用SiliconFlow API。",
            "episode_type": "text",
            "source_description": "测试脚本"
        }
    }
    
    print("🧪 测试添加记忆...")
    print(f"请求: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return True
        else:
            print(f"❌ 失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def test_search_memory():
    """测试搜索记忆"""
    url = "http://localhost:8000/tools/call"
    
    payload = {
        "name": "search",
        "arguments": {
            "query": "测试",
            "num_results": 3
        }
    }
    
    print("\n🔍 测试搜索记忆...")
    print(f"请求: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 搜索结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return True
        else:
            print(f"❌ 失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def main():
    print("🚀 开始测试Graphiti MCP服务器...")
    print("配置: gpt-4o-mini")
    print("-" * 50)
    
    # 测试添加记忆
    add_success = test_add_memory()
    
    if add_success:
        # 等待一下让记忆被处理
        print("\n⏳ 等待3秒让记忆被处理...")
        time.sleep(3)
        
        # 测试搜索记忆
        search_success = test_search_memory()
        
        if search_success:
            print("\n🎉 所有测试通过！Graphiti混合API配置工作正常！")
        else:
            print("\n⚠️ 搜索测试失败，但添加成功")
    else:
        print("\n❌ 添加记忆失败，可能是API速率限制")
        print("💡 建议等待几分钟后重试")

if __name__ == "__main__":
    main()