#!/usr/bin/env node
/**
 * 直接测试MCP工具调用响应格式
 */

const http = require('http');

function makeRequest(url, body) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const postData = JSON.stringify(body);
    
    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: urlObj.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
      }
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          const jsonData = JSON.parse(data);
          resolve({ status: res.statusCode, data: jsonData });
        } catch (e) {
          resolve({ status: res.statusCode, data: data, error: e.message });
        }
      });
    });

    req.on('error', (err) => reject(err));
    req.write(postData);
    req.end();
  });
}

async function testMCPResponse() {
  console.log('🧪 测试MCP工具调用响应格式...\n');

  try {
    // 测试搜索工具调用
    const searchRequest = {
      jsonrpc: '2.0',
      id: 'test-search',
      method: 'tools/call',
      params: {
        name: 'search',
        arguments: {
          query: '测试'
        }
      }
    };

    console.log('📤 发送请求:');
    console.log(JSON.stringify(searchRequest, null, 2));
    console.log('');

    const response = await makeRequest('http://localhost:8000/mcp/stream', searchRequest);
    
    console.log('📥 收到响应:');
    console.log(`状态码: ${response.status}`);
    console.log('响应体:');
    console.log(JSON.stringify(response.data, null, 2));
    
    if (response.data && response.data.result) {
      console.log('\n✅ 响应格式分析:');
      console.log(`- jsonrpc: ${response.data.jsonrpc}`);
      console.log(`- id: ${response.data.id}`);
      console.log(`- result类型: ${typeof response.data.result}`);
      console.log(`- result结构:`, Object.keys(response.data.result || {}));
      
      if (response.data.result.content) {
        console.log(`- content类型: ${typeof response.data.result.content}`);
        console.log(`- content长度: ${Array.isArray(response.data.result.content) ? response.data.result.content.length : 'N/A'}`);
      }
    }

  } catch (error) {
    console.log('❌ 测试失败:', error.message);
  }
}

testMCPResponse();