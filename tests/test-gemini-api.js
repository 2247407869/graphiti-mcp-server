#!/usr/bin/env node
/**
 * 测试Gemini API密钥是否有效
 */

const https = require('https');

const GEMINI_API_KEY = 'AIzaSyAOenGTsY7y_BZ6RzI_0QPU4n-N1eHAwKg';

function testGeminiAPI() {
  console.log('🧪 测试Gemini API密钥...\n');

  const data = JSON.stringify({
    contents: [{
      parts: [{
        text: "Hello, this is a test message."
      }]
    }]
  });

  const options = {
    hostname: 'generativelanguage.googleapis.com',
    port: 443,
    path: `/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length
    }
  };

  const req = https.request(options, (res) => {
    let responseData = '';
    
    console.log(`状态码: ${res.statusCode}`);
    console.log(`响应头:`, res.headers);
    
    res.on('data', (chunk) => {
      responseData += chunk;
    });
    
    res.on('end', () => {
      try {
        const jsonResponse = JSON.parse(responseData);
        if (res.statusCode === 200) {
          console.log('✅ Gemini API密钥有效');
          console.log('响应:', JSON.stringify(jsonResponse, null, 2));
        } else {
          console.log('❌ Gemini API调用失败');
          console.log('错误响应:', JSON.stringify(jsonResponse, null, 2));
        }
      } catch (e) {
        console.log('❌ 响应解析失败');
        console.log('原始响应:', responseData);
      }
    });
  });

  req.on('error', (error) => {
    console.log('❌ 请求失败:', error.message);
  });

  req.write(data);
  req.end();
}

testGeminiAPI();