#!/usr/bin/env python3
"""
检查Graphiti演示项目的完整状态
"""

import requests
import subprocess
import json
import time

def check_docker_services():
    """检查Docker服务状态"""
    print("🐳 检查Docker服务状态...")
    try:
        result = subprocess.run(['docker-compose', 'ps', '--format', 'json'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            services = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        service = json.loads(line)
                        services.append(service)
                    except:
                        pass
            
            print(f"   发现 {len(services)} 个服务:")
            for service in services:
                name = service.get('Name', 'Unknown')
                state = service.get('State', 'Unknown')
                status = service.get('Status', 'Unknown')
                ports = service.get('Publishers', [])
                port_info = ', '.join([f"{p.get('PublishedPort', '?')}:{p.get('TargetPort', '?')}" for p in ports]) if ports else "无端口映射"
                
                status_icon = "✅" if state == "running" else "❌"
                print(f"   {status_icon} {name}: {state} ({status}) - {port_info}")
            
            return len([s for s in services if s.get('State') == 'running']) == len(services)
        else:
            print(f"   ❌ Docker命令失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ 检查Docker服务异常: {e}")
        return False

def check_service_health(name, url, expected_keys=None):
    """检查单个服务健康状态"""
    print(f"🔍 检查 {name}...")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   ✅ {name} 正常运行")
                
                if expected_keys:
                    for key in expected_keys:
                        if key in data:
                            print(f"      {key}: {data[key]}")
                
                return True
            except:
                print(f"   ✅ {name} 正常运行 (非JSON响应)")
                return True
        else:
            print(f"   ❌ {name} 响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ {name} 连接失败")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ {name} 连接超时")
        return False
    except Exception as e:
        print(f"   ❌ {name} 检查异常: {e}")
        return False

def check_mcp_tools():
    """检查MCP工具功能"""
    print("🛠️  检查MCP工具功能...")
    try:
        # 检查工具列表
        response = requests.get("http://localhost:8000/tools/list", timeout=10)
        if response.status_code == 200:
            data = response.json()
            tools = data.get('tools', [])
            print(f"   ✅ 发现 {len(tools)} 个MCP工具:")
            for tool in tools:
                print(f"      - {tool['name']}: {tool['description']}")
            return len(tools) > 0
        else:
            print(f"   ❌ 工具列表获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ MCP工具检查异常: {e}")
        return False

def check_mcp_manifest():
    """检查MCP Manifest"""
    print("📋 检查MCP Manifest...")
    try:
        response = requests.get("http://localhost:8000/mcp/manifest", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Manifest正常")
            print(f"      插件名称: {data.get('name', 'Unknown')}")
            print(f"      版本: {data.get('version', 'Unknown')}")
            print(f"      工具数量: {len(data.get('tools', []))}")
            return True
        else:
            print(f"   ❌ Manifest获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Manifest检查异常: {e}")
        return False

def main():
    """主检查函数"""
    print("🚀 Graphiti演示项目状态检查")
    print("=" * 50)
    
    checks = [
        ("Docker服务", check_docker_services),
        ("FalkorDB", lambda: check_service_health("FalkorDB", "http://localhost:6379", None)),
        ("MCP服务器", lambda: check_service_health("MCP服务器", "http://localhost:8000/health", ["status", "mode", "graphiti_available"])),
        ("记忆管理界面", lambda: check_service_health("记忆管理界面", "http://localhost:3000", None)),
        ("LobeChat", lambda: check_service_health("LobeChat", "http://localhost:3210", None)),
        ("MCP工具", check_mcp_tools),
        ("MCP Manifest", check_mcp_manifest)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{'='*30}")
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name} 检查通过")
            else:
                print(f"❌ {check_name} 检查失败")
        except Exception as e:
            print(f"❌ {check_name} 检查异常: {e}")
        
        time.sleep(0.5)
    
    print(f"\n{'='*50}")
    print(f"检查总结: {passed}/{total} 通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有服务正常运行！")
        print("\n📋 下一步操作:")
        print("1. 访问 LobeChat: http://localhost:3210")
        print("2. 配置DeepSeek API密钥")
        print("3. 添加MCP插件:")
        print("   - 插件标识符: graphiti-memory")
        print("   - Manifest URL: http://graphiti-mcp:8000/mcp/manifest")
        print("4. 开始与具有记忆功能的AI助手对话！")
        
        print("\n🔗 服务链接:")
        print("- LobeChat聊天界面: http://localhost:3210")
        print("- 记忆管理界面: http://localhost:3000")
        print("- MCP API文档: http://localhost:8000")
        print("- 使用指南: http://localhost:3000/mcp-usage-guide.html")
        
    elif passed >= total * 0.7:
        print("⚠️  大部分服务正常，但有部分问题需要解决")
        print("\n🔧 建议操作:")
        print("1. 检查失败的服务日志: docker-compose logs [service-name]")
        print("2. 重启有问题的服务: docker-compose restart [service-name]")
        print("3. 如果问题持续，尝试重新构建: docker-compose up -d --build")
        
    else:
        print("❌ 多个服务存在问题，需要排查")
        print("\n🔧 故障排除:")
        print("1. 检查Docker是否正常运行")
        print("2. 查看所有服务日志: docker-compose logs")
        print("3. 重启所有服务: docker-compose down && docker-compose up -d")
        print("4. 检查端口占用情况")

if __name__ == "__main__":
    main()