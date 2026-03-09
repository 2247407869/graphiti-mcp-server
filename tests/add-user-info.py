#!/usr/bin/env python3
"""
添加用户信息到记忆中
"""

import requests
import json

def add_user_memory():
    """添加用户记忆信息"""
    print("📝 添加用户信息到记忆中...")
    
    # 用户基本信息
    user_info = {
        "name": "用户基本信息",
        "arguments": {
            "name": "用户基本信息",
            "episode_body": "用户的名字是李林松，是一名软件工程师，有7年工作经验。主要专注于后端开发，熟悉 Python 和 Java 编程语言。目前正在寻找新的工作机会。",
            "episode_type": "text",
            "source_description": "用户个人档案"
        }
    }
    
    # 技能信息
    skills_info = {
        "name": "用户技能信息",
        "arguments": {
            "name": "用户技能信息", 
            "episode_body": "李林松的技术技能包括：后端开发、Python编程、Java编程、数据库设计、API开发、微服务架构。他有丰富的软件开发经验。",
            "episode_type": "text",
            "source_description": "用户技能档案"
        }
    }
    
    # 当前状态
    current_status = {
        "name": "用户当前状态",
        "arguments": {
            "name": "用户当前状态",
            "episode_body": "李林松目前正在积极求职中，寻找Java高级开发工程师或Python后端开发的职位。他希望找到一个能发挥自己技术能力的工作环境。",
            "episode_type": "text", 
            "source_description": "用户状态更新"
        }
    }
    
    memories = [user_info, skills_info, current_status]
    
    for memory in memories:
        try:
            response = requests.post(
                "http://localhost:8000/tools/call",
                headers={"Content-Type": "application/json"},
                json=memory,
                timeout=30  # 增加超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 成功添加: {memory['name']}")
                episode_id = result.get('result', {}).get('episode_id')
                if episode_id:
                    print(f"   Episode ID: {episode_id}")
            else:
                print(f"❌ 添加失败: {memory['name']} - {response.status_code}")
                print(f"   响应: {response.text}")
                
        except Exception as e:
            print(f"❌ 添加异常: {memory['name']} - {e}")

def test_memory_search():
    """测试记忆搜索"""
    print("\n🔍 测试记忆搜索...")
    
    search_queries = [
        "李林松",
        "软件工程师", 
        "Python Java",
        "求职"
    ]
    
    for query in search_queries:
        try:
            search_data = {
                "name": "search",
                "arguments": {
                    "query": query,
                    "num_results": 3
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
                print(f"\n🔍 搜索 '{query}': 找到 {len(memories)} 个结果")
                
                for i, memory in enumerate(memories):
                    print(f"   {i+1}. {memory.get('name', 'N/A')}")
                    print(f"      内容: {memory.get('content', 'N/A')[:100]}...")
                    print(f"      相似度: {memory.get('score', 'N/A')}")
            else:
                print(f"❌ 搜索失败 '{query}': {response.status_code}")
                
        except Exception as e:
            print(f"❌ 搜索异常 '{query}': {e}")

def main():
    """主函数"""
    print("🧠 添加用户记忆信息")
    print("="*40)
    
    # 添加用户记忆
    add_user_memory()
    
    # 测试搜索
    test_memory_search()
    
    print("\n" + "="*40)
    print("✅ 用户信息已添加到记忆中！")
    print("\n💡 现在你可以在 LobeChat 中测试:")
    print("   - '你还记得我的名字吗？'")
    print("   - '我是做什么工作的？'")
    print("   - '我熟悉哪些编程语言？'")
    print("   - '我目前的状态是什么？'")

if __name__ == "__main__":
    main()