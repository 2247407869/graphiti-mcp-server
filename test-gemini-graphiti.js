#!/usr/bin/env node
/**
 * 测试Gemini版本的Graphiti MCP服务器
 */

const http = require('http');

async function testAddEpisode() {
  console.log('🧪 测试添加Episode到Graphiti（Gemini版本）...\n');

  const data = JSON.stringify({
    name: "add_episode",
    arguments: {
      name: "测试记忆",
      episode_body: "这是一个使用Gemini API的测试记忆片段。今天我们成功配置了Graphiti使用Google Gemini API而不是OpenAI API。",
      episode_type: "text",
      source_description: "Gemini测试"
    }
  });

  const options = {
    hostname: 'localhost',
    port: 8000,
    path: '/tools/call',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length
    }
  };

  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      let responseData = '';
      
      console.log(`状态码: ${res.statusCode}`);
      
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      
      res.on('end', () => {
        try {
          const jsonResponse = JSON.parse(responseData);
          if (res.statusCode === 200) {
            console.log('✅ Episode添加成功');
            console.log('响应:', JSON.stringify(jsonResponse, null, 2));
            resolve(jsonResponse);
          } else {
            console.log('❌ Episode添加失败');
            console.log('错误响应:', JSON.stringify(jsonResponse, null, 2));
            reject(new Error(`HTTP ${res.statusCode}`));
          }
        } catch (e) {
          console.log('❌ 响应解析失败');
          console.log('原始响应:', responseData);
          reject(e);
        }
      });
    });

    req.on('error', (error) => {
      console.log('❌ 请求失败:', error.message);
      reject(error);
    });

    req.write(data);
    req.end();
  });
}

async function testSearch() {
  console.log('\n🔍 测试搜索Episode...\n');

  const data = JSON.stringify({
    name: "search",
    arguments: {
      query: "Gemini API",
      num_results: 3
    }
  });

  const options = {
    hostname: 'localhost',
    port: 8000,
    path: '/tools/call',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length
    }
  };

  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      let responseData = '';
      
      console.log(`状态码: ${res.statusCode}`);
      
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      
      res.on('end', () => {
        try {
          const jsonResponse = JSON.parse(responseData);
          if (res.statusCode === 200) {
            console.log('✅ 搜索成功');
            console.log('搜索结果:', JSON.stringify(jsonResponse, null, 2));
            resolve(jsonResponse);
          } else {
            console.log('❌ 搜索失败');
            console.log('错误响应:', JSON.stringify(jsonResponse, null, 2));
            reject(new Error(`HTTP ${res.statusCode}`));
          }
        } catch (e) {
          console.log('❌ 响应解析失败');
          console.log('原始响应:', responseData);
          reject(e);
        }
      });
    });

    req.on('error', (error) => {
      console.log('❌ 请求失败:', error.message);
      reject(error);
    });

    req.write(data);
    req.end();
  });
}

async function main() {
  try {
    // 测试添加Episode
    await testAddEpisode();
    
    // 等待一下让数据处理完成
    console.log('\n⏳ 等待数据处理...');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // 测试搜索
    await testSearch();
    
    console.log('\n🎉 所有测试完成！');
  } catch (error) {
    console.error('❌ 测试失败:', error.message);
  }
}

main();