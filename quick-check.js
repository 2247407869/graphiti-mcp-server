#!/usr/bin/env node
/**
 * 快速检查MCP配置状态
 */

const fs = require('fs');
const http = require('http');

const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

async function quickCheck() {
  log('🔍 快速检查MCP配置状态...', 'bold');
  
  let allGood = true;
  
  // 1. 检查配置文件
  const configFile = './data/lobechat/config.json';
  if (fs.existsSync(configFile)) {
    try {
      const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
      if (config.mcpServers && config.mcpServers['graphiti-memory']) {
        log('✅ LobeChat配置文件正常', 'green');
      } else {
        log('❌ LobeChat配置文件缺少MCP配置', 'red');
        allGood = false;
      }
    } catch (e) {
      log('❌ LobeChat配置文件格式错误', 'red');
      allGood = false;
    }
  } else {
    log('❌ LobeChat配置文件不存在', 'red');
    allGood = false;
  }
  
  // 2. 检查MCP服务器
  try {
    const response = await new Promise((resolve, reject) => {
      const req = http.get('http://localhost:8000/health', (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => resolve({ status: res.statusCode, data }));
      });
      req.on('error', reject);
      req.setTimeout(3000, () => reject(new Error('Timeout')));
    });
    
    if (response.status === 200) {
      log('✅ MCP服务器运行正常', 'green');
    } else {
      log('❌ MCP服务器响应异常', 'red');
      allGood = false;
    }
  } catch (e) {
    log('❌ MCP服务器无法访问', 'red');
    allGood = false;
  }
  
  // 3. 检查LobeChat
  try {
    const response = await new Promise((resolve, reject) => {
      const req = http.get('http://localhost:3210', (res) => {
        resolve({ status: res.statusCode });
      });
      req.on('error', reject);
      req.setTimeout(3000, () => reject(new Error('Timeout')));
    });
    
    if ([200, 302, 307].includes(response.status)) {
      log('✅ LobeChat可以访问', 'green');
    } else {
      log('❌ LobeChat访问异常', 'red');
      allGood = false;
    }
  } catch (e) {
    log('❌ LobeChat无法访问', 'red');
    allGood = false;
  }
  
  // 结果
  log('\n' + '='.repeat(40), 'bold');
  if (allGood) {
    log('🎉 所有检查通过！MCP配置正常', 'green');
    log('\n📋 可以开始使用:', 'blue');
    log('1. 访问: http://localhost:3210', 'blue');
    log('2. 配置DeepSeek API密钥', 'blue');
    log('3. 开始智能对话！', 'blue');
  } else {
    log('⚠️  发现问题，需要修复', 'yellow');
    log('\n🔧 修复命令:', 'blue');
    log('node auto-configure-lobechat.js', 'blue');
    log('docker-compose restart lobechat', 'blue');
  }
}

quickCheck().catch(console.error);