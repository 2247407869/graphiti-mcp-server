#!/usr/bin/env python3
"""
测试Episode搜索 - 尝试获取原始Episode而不是Edge
"""

import requests
import json

def test_episode_search():
    """测试Episode搜索"""
    print("🔍 测试Episode搜索...")
    
    # 尝试不同的搜索查询
    queries = [
        "调试测试",  # Episode名称
        "我的名字是李林松",  # 原始内容
        "软件工程师",  # 关键词
        "李林松"  # 人名
    ]
    
    for query in queries:
        print(f"\n🔍 搜索: '{query}'")
        
        search_data = {
            "name": "search",
            "arguments": {
                "query": query,
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
            memories = result.get('result', [])
            print(f"   找到 {len(memories)} 个结果")
            
            # 分析结果类型
            episode_count = 0
            edge_count = 0
            
            for memory in memories:
                if memory.get('episode_type') == 'edge':
                    edge_count += 1
                else:
                    episode_count += 1
                    print(f"   📄 Episode: {memory.get('name')} - {memory.get('content')[:50]}...")
            
            print(f"   📊 Episode: {episode_count}, Edge: {edge_count}")
            
        else:
            print(f"   ❌ 搜索失败: {response.status_code}")

def test_get_episodes():
    """测试获取Episodes列表"""
    print("\n📚 测试获取Episodes列表...")
    
    list_data = {
        "name": "get_episodes",
        "arguments": {
            "limit": 20
        }
    }
    
    response = requests.post(
        "http://localhost:8000/tools/call",
        headers={"Content-Type": "application/json"},
        json=list_data,
        timeout=15
    )
    
    if response.status_code == 200:
        result = response.json()
        episodes = result.get('result', [])
        print(f"✅ 获取到 {len(episodes)} 个Episodes")
        
        for i, episode in enumerate(episodes[:5]):  # 只显示前5个
            print(f"   {i+1}. {episode.get('name', 'N/A')}")
            print(f"      内容: {episode.get('content', 'N/A')[:100]}...")
    else:
        print(f"❌ 获取Episodes失败: {response.status_code}")
        print(f"响应: {response.text}")

if __name__ == "__main__":
    test_episode_search()
    test_get_episodes()