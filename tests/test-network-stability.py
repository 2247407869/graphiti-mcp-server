#!/usr/bin/env python3
"""
测试网络稳定性
"""

import requests
import time
import statistics
import socket
from datetime import datetime
import subprocess
import platform

def test_basic_connectivity():
    """测试基本网络连接"""
    print("🌐 测试基本网络连接...")
    
    test_sites = [
        ("百度", "https://www.baidu.com"),
        ("Google", "https://www.google.com"),
        ("GitHub", "https://github.com"),
        ("老张API", "https://api.laozhang.ai"),
        ("SiliconFlow", "https://api.siliconflow.cn")
    ]
    
    results = []
    
    for name, url in test_sites:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            status = "✅" if response.status_code < 400 else "⚠️"
            
            print(f"   {name}: {status} {response.status_code} ({latency:.0f}ms)")
            results.append((name, True, latency))
            
        except requests.exceptions.Timeout:
            print(f"   {name}: ❌ 超时")
            results.append((name, False, 10000))
        except Exception as e:
            print(f"   {name}: ❌ 错误 - {str(e)[:50]}")
            results.append((name, False, 10000))
    
    return results

def test_dns_resolution():
    """测试DNS解析"""
    print("\n🔍 测试DNS解析...")
    
    domains = [
        "api.laozhang.ai",
        "api.siliconflow.cn",
        "github.com",
        "google.com"
    ]
    
    results = []
    
    for domain in domains:
        try:
            start_time = time.time()
            ip = socket.gethostbyname(domain)
            end_time = time.time()
            
            dns_time = (end_time - start_time) * 1000
            print(f"   {domain}: ✅ {ip} ({dns_time:.0f}ms)")
            results.append((domain, True, dns_time))
            
        except Exception as e:
            print(f"   {domain}: ❌ 解析失败 - {e}")
            results.append((domain, False, 1000))
    
    return results

def test_ping_stability():
    """测试ping稳定性"""
    print("\n🏓 测试Ping稳定性...")
    
    hosts = [
        ("百度", "baidu.com"),
        ("Google", "google.com"),
        ("GitHub", "github.com")
    ]
    
    results = []
    
    for name, host in hosts:
        try:
            # 根据操作系统选择ping命令
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", "4", host]
            else:
                cmd = ["ping", "-c", "4", host]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # 解析ping结果（简化版）
                output = result.stdout
                if "平均" in output or "Average" in output.lower():
                    print(f"   {name}: ✅ 连通")
                    results.append((name, True))
                else:
                    print(f"   {name}: ✅ 连通")
                    results.append((name, True))
            else:
                print(f"   {name}: ❌ 不通")
                results.append((name, False))
                
        except Exception as e:
            print(f"   {name}: ❌ 测试失败 - {e}")
            results.append((name, False))
    
    return results

def test_api_stability():
    """测试API稳定性（多次请求）"""
    print("\n🔄 测试API稳定性（连续5次请求）...")
    
    api_tests = [
        ("老张API Embeddings", "https://api.laozhang.ai/v1/embeddings", {
            "model": "text-embedding-ada-002",
            "input": "test"
        }),
        ("SiliconFlow Chat", "https://api.siliconflow.cn/v1/chat/completions", {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10
        })
    ]
    
    results = []
    
    for name, url, payload in api_tests:
        print(f"\n   测试 {name}:")
        
        api_key_map = {
            "老张API": "sk-rmMS3NM1iiJI7BkzF153946dCaA4491a9cD73907F7001834",
            "SiliconFlow": "sk-gyowdkndmteuykdkamicbqdpcczdlmurlfdrcduyonoqtzwo"
        }
        
        # 选择API密钥
        api_key = None
        for key_name, key_value in api_key_map.items():
            if key_name in name:
                api_key = key_value
                break
        
        if not api_key:
            print(f"     ❌ 未找到API密钥")
            continue
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        success_count = 0
        latencies = []
        
        for i in range(5):
            try:
                start_time = time.time()
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000
                latencies.append(latency)
                
                if response.status_code == 200:
                    success_count += 1
                    print(f"     第{i+1}次: ✅ {response.status_code} ({latency:.0f}ms)")
                else:
                    print(f"     第{i+1}次: ⚠️ {response.status_code} ({latency:.0f}ms)")
                
            except requests.exceptions.Timeout:
                print(f"     第{i+1}次: ❌ 超时")
                latencies.append(30000)
            except Exception as e:
                print(f"     第{i+1}次: ❌ 错误 - {str(e)[:30]}")
                latencies.append(30000)
            
            # 请求间隔
            if i < 4:
                time.sleep(1)
        
        # 计算统计信息
        if latencies:
            avg_latency = statistics.mean(latencies)
            success_rate = (success_count / 5) * 100
            
            print(f"     📊 成功率: {success_rate:.0f}% | 平均延迟: {avg_latency:.0f}ms")
            results.append((name, success_rate, avg_latency))
        
    return results

def test_network_speed():
    """测试网络速度（简化版）"""
    print("\n⚡ 测试网络速度...")
    
    # 测试下载一个小文件
    test_urls = [
        ("GitHub", "https://github.com/robots.txt"),
        ("百度", "https://www.baidu.com/robots.txt")
    ]
    
    results = []
    
    for name, url in test_urls:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                size = len(response.content)
                duration = end_time - start_time
                speed = (size / duration) / 1024  # KB/s
                
                print(f"   {name}: ✅ {size} bytes in {duration:.2f}s ({speed:.1f} KB/s)")
                results.append((name, True, speed))
            else:
                print(f"   {name}: ❌ {response.status_code}")
                results.append((name, False, 0))
                
        except Exception as e:
            print(f"   {name}: ❌ 错误 - {str(e)[:50]}")
            results.append((name, False, 0))
    
    return results

def main():
    print("🚀 网络稳定性测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 执行各项测试
    connectivity_results = test_basic_connectivity()
    dns_results = test_dns_resolution()
    ping_results = test_ping_stability()
    api_results = test_api_stability()
    speed_results = test_network_speed()
    
    # 生成报告
    print("\n" + "=" * 60)
    print("📋 网络稳定性报告")
    print("=" * 60)
    
    # 基本连接性
    print("\n🌐 基本连接性:")
    successful_connections = sum(1 for _, success, _ in connectivity_results if success)
    print(f"   成功连接: {successful_connections}/{len(connectivity_results)} 个站点")
    
    # DNS解析
    print("\n🔍 DNS解析:")
    successful_dns = sum(1 for _, success, _ in dns_results if success)
    print(f"   成功解析: {successful_dns}/{len(dns_results)} 个域名")
    
    # Ping测试
    print("\n🏓 Ping测试:")
    successful_pings = sum(1 for _, success in ping_results if success)
    print(f"   成功Ping: {successful_pings}/{len(ping_results)} 个主机")
    
    # API稳定性
    print("\n🔄 API稳定性:")
    for name, success_rate, avg_latency in api_results:
        status = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 50 else "❌"
        print(f"   {name}: {status} {success_rate:.0f}% 成功率, {avg_latency:.0f}ms 平均延迟")
    
    # 网络速度
    print("\n⚡ 网络速度:")
    for name, success, speed in speed_results:
        if success:
            status = "✅" if speed > 100 else "⚠️" if speed > 10 else "❌"
            print(f"   {name}: {status} {speed:.1f} KB/s")
        else:
            print(f"   {name}: ❌ 测试失败")
    
    # 总体评估
    print("\n" + "=" * 60)
    print("🎯 总体评估:")
    
    # 计算总体得分
    connectivity_score = (successful_connections / len(connectivity_results)) * 100
    dns_score = (successful_dns / len(dns_results)) * 100
    ping_score = (successful_pings / len(ping_results)) * 100
    
    if api_results:
        api_score = statistics.mean([rate for _, rate, _ in api_results])
    else:
        api_score = 0
    
    overall_score = statistics.mean([connectivity_score, dns_score, ping_score, api_score])
    
    if overall_score >= 80:
        print(f"   网络状态: ✅ 良好 ({overall_score:.0f}分)")
        print("   💡 网络连接稳定，适合API调用")
    elif overall_score >= 60:
        print(f"   网络状态: ⚠️ 一般 ({overall_score:.0f}分)")
        print("   💡 网络有些不稳定，建议增加重试机制")
    else:
        print(f"   网络状态: ❌ 较差 ({overall_score:.0f}分)")
        print("   💡 网络连接不稳定，建议检查网络配置")
    
    # 针对老张API的建议
    laozhang_results = [r for r in api_results if "老张API" in r[0]]
    if laozhang_results:
        laozhang_success_rate = laozhang_results[0][1]
        laozhang_latency = laozhang_results[0][2]
        
        print(f"\n🎯 老张API专项分析:")
        print(f"   成功率: {laozhang_success_rate:.0f}%")
        print(f"   平均延迟: {laozhang_latency:.0f}ms")
        
        if laozhang_success_rate >= 80 and laozhang_latency < 5000:
            print("   💡 老张API连接良好，可以正常使用")
        elif laozhang_success_rate >= 60:
            print("   💡 老张API连接一般，建议增加超时时间和重试")
        else:
            print("   💡 老张API连接不稳定，建议检查网络或考虑备用方案")

if __name__ == "__main__":
    main()