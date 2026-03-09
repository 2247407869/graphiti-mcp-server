#!/usr/bin/env node
/**
 * 测试MCP代理脚本
 */

const { spawn } = require('child_process');
const path = require('path');

function testMCPProxy() {
    console.log('🧪 测试MCP代理...');
    
    // 启动MCP代理进程
    const proxy = spawn('node', [path.join(__dirname, 'config/lobechat/mcp-proxy.js')], {
        env: {
            ...process.env,
            MCP_SERVER_URL: 'http://localhost:8000'
        }
    });

    let responseBuffer = '';

    proxy.stdout.on('data', (data) => {
        responseBuffer += data.toString();
        console.log('📤 代理响应:', data.toString().trim());
    });

    proxy.stderr.on('data', (data) => {
        console.error('❌ 代理错误:', data.toString());
    });

    // 发送初始化请求
    setTimeout(() => {
        console.log('📨 发送初始化请求...');
        const initRequest = {
            jsonrpc: "2.0",
            id: 1,
            method: "initialize",
            params: {
                protocolVersion: "2024-11-05",
                capabilities: {},
                clientInfo: {
                    name: "test-client",
                    version: "1.0.0"
                }
            }
        };
        proxy.stdin.write(JSON.stringify(initRequest) + '\n');
    }, 1000);

    // 发送工具列表请求
    setTimeout(() => {
        console.log('📨 发送工具列表请求...');
        const toolsRequest = {
            jsonrpc: "2.0",
            id: 2,
            method: "tools/list",
            params: {}
        };
        proxy.stdin.write(JSON.stringify(toolsRequest) + '\n');
    }, 2000);

    // 发送工具调用请求
    setTimeout(() => {
        console.log('📨 发送工具调用请求...');
        const callRequest = {
            jsonrpc: "2.0",
            id: 3,
            method: "tools/call",
            params: {
                name: "add_episode",
                arguments: {
                    name: "MCP测试记忆",
                    episode_body: "这是通过MCP代理添加的测试记忆"
                }
            }
        };
        proxy.stdin.write(JSON.stringify(callRequest) + '\n');
    }, 3000);

    // 5秒后结束测试
    setTimeout(() => {
        console.log('✅ 测试完成');
        proxy.kill();
        process.exit(0);
    }, 5000);
}

testMCPProxy();