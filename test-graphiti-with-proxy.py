#!/usr/bin/env python3
"""
测试通过API代理的Graphiti功能
"""

import requests
import json
import time

def test_add_memory():
    """测试添加记忆功能"""
    print("🧪 测试添加记忆功能...")
    
    url = "http://localhost:8000/tools/call"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "name": "add_episode",
        "arguments": {
            "name": "API代理测试记忆",
            "episode_body": "这是一个通过API代理测试的记忆片段。我们验证SiliconFlow API通过代理转换后能否正常工作。",
            "episode_type": "text",
            "source_description": "API代理测试"
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 添加记忆成功！")
            print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 检查是否真的成功
            if result.get("result", {}).get("success"):
                episode_id = result.get("result", {}).get("episode_id")
                print(f"🎯 记忆ID: {episode_id}")
                return True, episode_id
            else:
                print(f"❌ 添加失败: {result.get('result', {}).get('error', 'Unknown error')}")
                return False, None
        else:
            print(f"❌ 请求失败: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False, None

def test_search_memory():
    """测试搜索记忆功能"""
    print("\n🧪 测试搜索记忆功能...")
    
    url = "http://localhost:8000/tools/call"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "name": "search",
        "arguments": {
            "query": "API代理测试",
            "num_results": 3
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 搜索记忆成功！")
            
            search_results = result.get("result", [])
            print(f"🔍 找到 {len(search_results)} 个结果:")
            
            for i, item in enumerate(search_results):
                print(f"   {i+1}. ID: {item.get('id', 'unknown')}")
                print(f"      名称: {item.get('name', 'Unnamed')}")
                print(f"      内容: {item.get('content', '')[:100]}...")
                print(f"      分数: {item.get('score', 0)}")
            
            return True, search_results
        else:
            print(f"❌ 搜索失败: {response.text}")
            return False, []
            
    except Exception as e:
        print(f"❌ 搜索异常: {e}")
        return False, []

def test_get_episodes():
    """测试获取记忆列表"""
    print("\n🧪 测试获取记忆列表...")
    
    url = "http://localhost:8000/tools/call"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "name": "get_episodes",
        "arguments": {
            "limit": 10
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 获取记忆列表成功！")
            
            episodes = result.get("result", [])
            print(f"📋 共有 {len(episodes)} 个记忆:")
            
            for i, episode in enumerate(episodes):
                print(f"   {i+1}. ID: {episode.get('id', 'unknown')}")
                print(f"      名称: {episode.get('name', 'Unnamed')}")
                print(f"      创建时间: {episode.get('created_at', 'Unknown')}")
            
            return True, episodes
        else:
            print(f"❌ 获取列表失败: {response.text}")
            return False, []
            
    except Exception as e:
        print(f"❌ 获取列表异常: {e}")
        return False, []

def main():
    """主测试函数"""
    print("🚀 测试通过API代理的Graphiti功能...")
    print("=" * 60)
    
    results = []
    
    # 测试添加记忆
    add_success, episode_id = test_add_memory()
    results.append(("添加记忆", add_success))
    
    if add_success:
        # 等待一下让记忆处理完成
        print("⏳ 等待记忆处理...")
        time.sleep(3)
        
        # 测试搜索记忆
        search_success, search_results = test_search_memory()
        results.append(("搜索记忆", search_success))
        
        # 测试获取记忆列表
        list_success, episodes = test_get_episodes()
        results.append(("获取记忆列表", list_success))
    
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
        print("🎉 所有测试通过！")
        print("✅ SiliconFlow API通过代理成功集成到Graphiti")
        print("✅ 记忆添加、搜索、列表功能都正常工作")
        print("💡 现在可以在LobeChat中使用MCP工具了")
    else:
        print("⚠️  部分测试失败")
        if not add_success:
            print("💡 添加记忆失败，可能是LLM推理问题")
        else:
            print("💡 基本功能正常，部分高级功能可能需要调试")
    
    return all_passed

if __name__ == "__main__":
    main()