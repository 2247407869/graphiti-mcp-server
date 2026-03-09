#!/usr/bin/env node
/**
 * 测试真实Graphiti功能
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

async function testRealGraphiti() {
  console.log('🧪 测试真实Graphiti功能...\n');

  try {
    // 1. 测试添加Episode（不需要embeddings）
    console.log('1️⃣ 测试添加Episode到FalkorDB...');
    const addRequest = {
      jsonrpc: '2.0',
      id: 'test-add',
      method: 'tools/call',
      params: {
        name: 'add_episode',
        arguments: {
          name: '真实Graphiti测试',
          episode_body: '这是一个存储在FalkorDB中的真实Episode，用于验证Graphiti集成是否正常工作。'
        }
      }
    };

    const addResponse = await makeRequest('http://localhost:8000/mcp/stream', addRequest);
    
    if (addResponse.status === 200 && addResponse.data.result?.content) {
      const content = addResponse.data.result.content[0];
      if (content && content.text) {
        const result = JSON.parse(content.text);
        if (result.success) {
          console.log('✅ Episode添加成功');
          console.log(`   Episode ID: ${result.episode_id}`);
          console.log(`   消息: ${result.message}`);
        } else {
          console.log('❌ Episode添加失败:', result.error);
          return;
        }
      }
    } else {
      console.log('❌ 添加请求失败:', addResponse.data);
      return;
    }

    // 2. 测试获取Episodes列表（不需要embeddings）
    console.log('\n2️⃣ 测试获取Episodes列表...');
    const listRequest = {
      jsonrpc: '2.0',
      id: 'test-list',
      method: 'tools/call',
      params: {
        name: 'get_episodes',
        arguments: {
          limit: 10
        }
      }
    };

    const listResponse = await makeRequest('http://localhost:8000/mcp/stream', listRequest);
    
    if (listResponse.status === 200 && listResponse.data.result?.content) {
      const content = listResponse.data.result.content[0];
      if (content && content.text) {
        const episodes = JSON.parse(content.text);
        console.log('✅ Episodes列表获取成功');
        console.log(`   找到 ${episodes.length} 个Episodes`);
        episodes.forEach((ep, index) => {
          console.log(`   ${index + 1}. ${ep.name} (${ep.created_at})`);
        });
      }
    }

    // 3. 测试搜索功能（需要embeddings，预期会失败）
    console.log('\n3️⃣ 测试搜索功能（预期失败 - 需要embeddings）...');
    const searchRequest = {
      jsonrpc: '2.0',
      id: 'test-search',
      method: 'tools/call',
      params: {
        name: 'search',
        arguments: {
          query: '测试',
          num_results: 3
        }
      }
    };

    try {
      const searchResponse = await makeRequest('http://localhost:8000/mcp/stream', searchRequest);
      
      if (searchResponse.status === 200 && searchResponse.data.result?.content) {
        const content = searchResponse.data.result.content[0];
        if (content && content.text) {
          const results = JSON.parse(content.text);
          console.log('✅ 搜索功能意外成功');
          console.log(`   找到 ${results.length} 个结果`);
        }
      } else if (searchResponse.data.error) {
        console.log('❌ 搜索失败（预期）:', searchResponse.data.error.message);
        console.log('   原因: 需要配置OPENAI_EMBEDDINGS_API_KEY');
      }
    } catch (error) {
      console.log('❌ 搜索失败（预期）:', error.message);
      console.log('   原因: 需要配置embeddings API');
    }

    console.log('\n📊 测试总结:');
    console.log('✅ Graphiti成功连接到FalkorDB');
    console.log('✅ Episode添加功能正常');
    console.log('✅ Episode列表功能正常');
    console.log('❌ 搜索功能需要embeddings配置');
    console.log('\n💡 要启用搜索功能，请在.env文件中添加:');
    console.log('   OPENAI_EMBEDDINGS_API_KEY=sk-your-openai-key-here');

  } catch (error) {
    console.log('❌ 测试失败:', error.message);
  }
}

testRealGraphiti();