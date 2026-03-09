#!/usr/bin/env python3
"""
调试搜索功能
"""

import requests
import json

def debug_search():
    """调试搜索功能"""
    print("🔍 调试搜索功能...")
    
    # 先添加一个简单的记忆
    print("\n1. 添加测试记忆...")
    add_data = {
        "name": "add_episode",
        "arguments": {
            "name": "调试测试",
            "episode_body": "我的名字是李林松，我是一名软件工程师。",
            "episode_type": "text",
            "source_description": "调试测试"
        }
    }
    
    response = requests.post(
        "http://localhost:8000/tools/call",
        headers={"Content-Type": "application/json"},
        json=add_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 添加成功")
        print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    else:
        print(f"❌ 添加失败: {response.status_code}")
        print(f"响应: {response.text}")
        return
    
    # 等待一下让索引更新
    import time
    print("\n2. 等待索引更新...")
    time.sleep(3)
    
    # 搜索
    print("\n3. 搜索测试...")
    search_data = {
        "name": "search",
        "arguments": {
            "query": "李林松",
            "num_results": 10
        }
    }
    
    response = requests.post(
        "http://localhost:8000/tools/call",
        headers={"Content-Type": "application/json"},
        json=search_data,
        timeout=15
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 搜索成功")
        print(f"搜索结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        # 分析结果
        memories = result.get('result', [])
        print(f"\n📊 找到 {len(memories)} 个结果:")
        for i, memory in enumerate(memories):
            print(f"\n结果 {i+1}:")
            print(f"  ID: {memory.get('id')}")
            print(f"  名称: {memory.get('name')}")
            print(f"  内容: {memory.get('content')}")
            print(f"  类型: {memory.get('episode_type')}")
            print(f"  相似度: {memory.get('score')}")
            
    else:
        print(f"❌ 搜索失败: {response.status_code}")
        print(f"响应: {response.text}")

if __name__ == "__main__":
    debug_search()