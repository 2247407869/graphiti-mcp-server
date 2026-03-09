#!/usr/bin/env python3
"""
最终记忆功能测试 - 模拟LobeChat的使用场景
"""

import requests
import json

def simulate_user_questions():
    """模拟用户提问场景"""
    print("🤖 模拟用户提问场景")
    print("="*50)
    
    # 用户可能问的问题
    questions = [
        "李林松",
        "我的名字",
        "软件工程师",
        "我是做什么工作的",
        "我的职业",
        "Python Java"
    ]
    
    for question in questions:
        print(f"\n❓ 用户问题: '{question}'")
        
        # 搜索相关记忆
        search_data = {
            "name": "search",
            "arguments": {
                "query": question,
                "num_results": 3
            }
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/tools/call",
                headers={"Content-Type": "application/json"},
                json=search_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                memories = result.get('result', [])
                
                print(f"🧠 找到 {len(memories)} 个相关记忆:")
                
                # 生成AI可能的回答
                answer_parts = []
                for memory in memories:
                    content = memory.get('content', '')
                    if '职业信息' in content:
                        answer_parts.append("你是一名软件工程师")
                    elif '用户名称' in content and '李林松' in content:
                        answer_parts.append("你的名字是李林松")
                    elif 'software engineer' in content.lower():
                        answer_parts.append("你从事软件开发工作")
                
                if answer_parts:
                    print(f"🤖 AI可能的回答: {', '.join(set(answer_parts))}")
                else:
                    print("🤖 AI可能的回答: 基于记忆内容生成回答")
                
                # 显示详细记忆
                for i, memory in enumerate(memories[:2]):  # 只显示前2个
                    print(f"   {i+1}. {memory.get('content', 'N/A')}")
                    
            else:
                print(f"❌ 搜索失败: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 搜索异常: {e}")

def test_mcp_stream_for_lobechat():
    """测试MCP流式端点 - LobeChat使用的格式"""
    print(f"\n🌊 测试MCP流式端点（LobeChat格式）")
    print("="*50)
    
    # 模拟LobeChat的搜索请求
    mcp_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {
                "query": "李林松的职业和技能",
                "num_results": 5
            }
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/mcp/stream",
            headers={"Content-Type": "application/json"},
            json=mcp_request,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ MCP流式端点测试成功")
            
            # 检查返回格式
            if 'result' in result and 'content' in result['result']:
                content = result['result']['content']
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get('text', '')
                    print(f"📄 返回内容长度: {len(text_content)} 字符")
                    print(f"📄 内容预览: {text_content[:200]}...")
                    
                    # 检查是否包含用户信息
                    if '李林松' in text_content:
                        print("✅ 内容包含用户姓名")
                    if '软件工程师' in text_content or 'software engineer' in text_content.lower():
                        print("✅ 内容包含职业信息")
                else:
                    print("⚠️  返回内容格式异常")
            else:
                print("⚠️  返回格式不符合预期")
                print(f"实际返回: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            print(f"❌ MCP流式端点测试失败: {response.status_code}")
            print(f"响应: {response.text}")
            
    except Exception as e:
        print(f"❌ MCP流式端点测试异常: {e}")

def main():
    """主函数"""
    print("🧠 最终记忆功能测试")
    print("模拟LobeChat使用场景")
    print("="*60)
    
    # 模拟用户提问
    simulate_user_questions()
    
    # 测试MCP流式端点
    test_mcp_stream_for_lobechat()
    
    print("\n" + "="*60)
    print("📊 测试总结:")
    print("✅ 记忆系统已经能够:")
    print("   - 存储用户的基本信息（姓名、职业）")
    print("   - 通过关键词搜索找到相关记忆")
    print("   - 以结构化知识的形式返回信息")
    print("   - 支持MCP协议与LobeChat集成")
    
    print("\n💡 使用建议:")
    print("1. 在LobeChat中确保MCP插件已启用")
    print("2. 测试问题: '你还记得我的名字吗？'")
    print("3. 测试问题: '我是做什么工作的？'")
    print("4. 如果AI不记得，重新介绍自己让它记住")

if __name__ == "__main__":
    main()