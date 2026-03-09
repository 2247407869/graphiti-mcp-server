#!/usr/bin/env node
/**
 * 添加测试记忆数据
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

async function addTestMemories() {
  console.log('📝 添加测试记忆数据...\n');

  const memories = [
    {
      name: '用户基本信息',
      content: '用户是李林松，Java高级开发工程师，有7年工作经验，专长Spring Boot和微服务架构'
    },
    {
      name: '技术偏好',
      content: '用户偏好使用TypeScript而不是JavaScript，喜欢详细的技术文档，要求代码简洁'
    },
    {
      name: '工作习惯',
      content: '用户经常使用快捷键操作，偏好深色主题界面，注重代码质量和性能优化'
    }
  ];

  for (let i = 0; i < memories.length; i++) {
    const memory = memories[i];
    console.log(`${i + 1}. 添加记忆: ${memory.name}`);
    
    const addRequest = {
      jsonrpc: '2.0',
      id: `add-${i + 1}`,
      method: 'tools/call',
      params: {
        name: 'add_episode',
        arguments: {
          name: memory.name,
          episode_body: memory.content
        }
      }
    };

    try {
      const response = await makeRequest('http://localhost:8000/mcp/stream', addRequest);
      
      if (response.status === 200) {
        console.log('   ✅ 添加成功');
        if (response.data.result && response.data.result.content) {
          const content = response.data.result.content[0];
          if (content && content.text) {
            const result = JSON.parse(content.text);
            console.log(`   📝 Episode ID: ${result.episode_id}`);
          }
        }
      } else {
        console.log('   ❌ 添加失败:', response.data);
      }
    } catch (error) {
      console.log('   ❌ 请求失败:', error.message);
    }
    
    console.log('');
  }

  // 测试搜索
  console.log('🔍 测试搜索功能...');
  const searchRequest = {
    jsonrpc: '2.0',
    id: 'search-test',
    method: 'tools/call',
    params: {
      name: 'search',
      arguments: {
        query: '用户',
        num_results: 5
      }
    }
  };

  try {
    const response = await makeRequest('http://localhost:8000/mcp/stream', searchRequest);
    
    console.log('📥 搜索响应:');
    console.log(JSON.stringify(response.data, null, 2));
    
    if (response.data.result && response.data.result.content) {
      const content = response.data.result.content[0];
      if (content && content.text) {
        const results = JSON.parse(content.text);
        console.log(`\n✅ 找到 ${results.length} 个相关记忆`);
        results.forEach((result, index) => {
          console.log(`   ${index + 1}. ${result.name} (相似度: ${result.score})`);
        });
      }
    }
  } catch (error) {
    console.log('❌ 搜索失败:', error.message);
  }
}

addTestMemories();