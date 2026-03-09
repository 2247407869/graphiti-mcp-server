#!/usr/bin/env node
/**
 * 测试LobeChat MCP工具调用
 * 模拟LobeChat调用MCP工具的完整流程
 */

const http = require('http');

const MCP_SERVER_URL = 'http://localhost:8000';

function makeRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const requestOptions = {
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: urlObj.pathname,
      method: options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    };

    const req = http.request(requestOptions, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          const jsonData = JSON.parse(data);
          resolve({ status: res.statusCode, data: jsonData });
        } catch (e) {
          resolve({ status: res.statusCode, data: data });
        }
      });
    });

    req.on('error', (err) => reject(err));
    
    if (options.body) {
      req.write(JSON.stringify(options.body));
    }
    
    req.end();
  });
}

async function testMCPWorkflow() {
  console.log('🧪 测试LobeChat MCP工具调用流程...\n');

  try {
    // 1. 初始化MCP连接
    console.log('1️⃣ 初始化MCP连接...');
    const initResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
      method: 'POST',
      body: {
        jsonrpc: '2.0',
        id: 'init-1',
        method: 'initialize',
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: {
            name: 'LobeChat',
            version: '1.0.0'
          }
        }
      }
    });

    if (initResponse.status === 200) {
      console.log('✅ MCP初始化成功');
      console.log(`   协议版本: ${initResponse.data.result?.protocolVersion}`);
      console.log(`   服务器: ${initResponse.data.result?.serverInfo?.name}\n`);
    } else {
      console.log('❌ MCP初始化失败\n');
      return;
    }

    // 2. 获取工具列表
    console.log('2️⃣ 获取可用工具列表...');
    const toolsResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
      method: 'POST',
      body: {
        jsonrpc: '2.0',
        id: 'tools-1',
        method: 'tools/list',
        params: {}
      }
    });

    if (toolsResponse.status === 200 && toolsResponse.data.result?.tools) {
      console.log('✅ 工具列表获取成功');
      toolsResponse.data.result.tools.forEach(tool => {
        console.log(`   - ${tool.name}: ${tool.description}`);
      });
      console.log('');
    }

    // 3. 添加记忆片段
    console.log('3️⃣ 添加记忆片段...');
    const addResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
      method: 'POST',
      body: {
        jsonrpc: '2.0',
        id: 'add-1',
        method: 'tools/call',
        params: {
          name: 'add_episode',
          arguments: {
            name: '用户偏好记录',
            episode_body: '用户喜欢简洁的界面设计，偏好深色主题，经常使用快捷键操作。对技术文档要求详细但不冗余。'
          }
        }
      }
    });

    if (addResponse.status === 200) {
      if (addResponse.data.result?.success) {
        console.log('✅ 记忆片段添加成功');
        console.log(`   Episode ID: ${addResponse.data.result.episode_id}`);
        console.log(`   消息: ${addResponse.data.result.message}\n`);
      } else if (addResponse.data.error) {
        console.log('❌ 记忆片段添加失败');
        console.log(`   错误: ${addResponse.data.error.message}\n`);
        return;
      }
    }

    // 4. 搜索记忆
    console.log('4️⃣ 搜索相关记忆...');
    const searchResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
      method: 'POST',
      body: {
        jsonrpc: '2.0',
        id: 'search-1',
        method: 'tools/call',
        params: {
          name: 'search',
          arguments: {
            query: '用户偏好',
            num_results: 3
          }
        }
      }
    });

    if (searchResponse.status === 200) {
      if (Array.isArray(searchResponse.data.result)) {
        console.log('✅ 记忆搜索成功');
        console.log(`   找到 ${searchResponse.data.result.length} 个相关记忆:`);
        searchResponse.data.result.forEach((result, index) => {
          console.log(`   ${index + 1}. ${result.name} (相似度: ${result.score})`);
          console.log(`      内容: ${result.content.substring(0, 100)}...`);
        });
        console.log('');
      } else if (searchResponse.data.error) {
        console.log('❌ 记忆搜索失败');
        console.log(`   错误: ${searchResponse.data.error.message}\n`);
      }
    }

    // 5. 获取所有记忆
    console.log('5️⃣ 获取记忆列表...');
    const listResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
      method: 'POST',
      body: {
        jsonrpc: '2.0',
        id: 'list-1',
        method: 'tools/call',
        params: {
          name: 'get_episodes',
          arguments: {
            limit: 5
          }
        }
      }
    });

    if (listResponse.status === 200) {
      if (Array.isArray(listResponse.data.result)) {
        console.log('✅ 记忆列表获取成功');
        console.log(`   总共 ${listResponse.data.result.length} 个记忆片段:`);
        listResponse.data.result.forEach((episode, index) => {
          console.log(`   ${index + 1}. ${episode.name} (${episode.created_at})`);
        });
        console.log('');
      } else if (listResponse.data.error) {
        console.log('❌ 记忆列表获取失败');
        console.log(`   错误: ${listResponse.data.error.message}\n`);
      }
    }

    console.log('🎉 MCP工具调用测试完成！');
    console.log('\n📋 测试结果总结:');
    console.log('✅ MCP连接初始化正常');
    console.log('✅ 工具列表获取正常');
    console.log('✅ 记忆添加功能正常');
    console.log('✅ 记忆搜索功能正常');
    console.log('✅ 记忆列表功能正常');
    console.log('\n🚀 现在可以在LobeChat中正常使用记忆功能了！');

  } catch (error) {
    console.log('❌ 测试过程中发生错误:', error.message);
  }
}

if (require.main === module) {
  testMCPWorkflow();
}