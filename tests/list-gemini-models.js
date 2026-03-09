#!/usr/bin/env node
/**
 * 列出Gemini API可用的模型
 */

const https = require('https');

const GEMINI_API_KEY = 'AIzaSyAOenGTsY7y_BZ6RzI_0QPU4n-N1eHAwKg';

function listGeminiModels() {
  console.log('📋 列出Gemini API可用模型...\n');

  const options = {
    hostname: 'generativelanguage.googleapis.com',
    port: 443,
    path: `/v1beta/models?key=${GEMINI_API_KEY}`,
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
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
          console.log('✅ 成功获取模型列表');
          console.log('可用模型:');
          if (jsonResponse.models) {
            jsonResponse.models.forEach(model => {
              console.log(`- ${model.name}`);
              if (model.supportedGenerationMethods) {
                console.log(`  支持的方法: ${model.supportedGenerationMethods.join(', ')}`);
              }
            });
          }
        } else {
          console.log('❌ 获取模型列表失败');
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

  req.end();
}

listGeminiModels();