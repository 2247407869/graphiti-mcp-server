#!/usr/bin/env node
/**
 * 测试Gemini embeddings API
 */

const https = require('https');

const GEMINI_API_KEY = 'AIzaSyAOenGTsY7y_BZ6RzI_0QPU4n-N1eHAwKg';

function testGeminiEmbeddings() {
  console.log('🧪 测试Gemini embeddings API...\n');

  const data = JSON.stringify({
    model: "models/text-embedding-004",
    content: {
      parts: [{
        text: "This is a test sentence for embedding."
      }]
    }
  });

  const options = {
    hostname: 'generativelanguage.googleapis.com',
    port: 443,
    path: `/v1beta/models/text-embedding-004:embedContent?key=${GEMINI_API_KEY}`,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length
    }
  };

  const req = https.request(options, (res) => {
    let responseData = '';
    
    console.log(`状态码: ${res.statusCode}`);
    
    res.on('data', (chunk) => {
      responseData += chunk;
    });
    
    res.on('end', () => {
      try {
        const jsonResponse = JSON.parse(responseData);
        if (res.statusCode === 200) {
          console.log('✅ Gemini embeddings API工作正常');
          console.log('Embedding维度:', jsonResponse.embedding?.values?.length || 'unknown');
          console.log('前5个值:', jsonResponse.embedding?.values?.slice(0, 5) || 'none');
        } else {
          console.log('❌ Gemini embeddings API调用失败');
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

testGeminiEmbeddings();