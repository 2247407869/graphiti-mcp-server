#!/usr/bin/env node
/**
 * 检查LobeChat界面中的MCP插件状态
 * 通过模拟浏览器请求来检查插件是否正确加载
 */

const http = require('http');
const https = require('https');

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

async function makeRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const lib = urlObj.protocol === 'https:' ? https : http;
    
    const requestOptions = {
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: urlObj.pathname + (urlObj.search || ''),
      method: options.method || 'GET',
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        ...options.headers
      }
    };

    const req = lib.request(requestOptions, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        resolve({ 
          status: res.statusCode, 
          headers: res.headers,
          data: data 
        });
      });
    });

    req.on('error', (err) => reject(err));
    req.setTimeout(10000, () => reject(new Error('Request timeout')));
    
    if (options.body) {
      req.write(options.body);
    }
    
    req.end();
  });
}

async function checkLobeChatAPI() {
  log('\n🔍 检查LobeChat API端点...', 'blue');
  
  try {
    // 检查LobeChat的API配置端点
    const apiEndpoints = [
      '/api/config',
      '/api/plugins',
      '/api/mcp',
      '/api/settings'
    ];
    
    for (const endpoint of apiEndpoints) {
      try {
        const response = await makeRequest(`http://localhost:3210${endpoint}`);
        log(`   ${endpoint}: ${response.status}`, response.status === 200 ? 'green' : 'yellow');
        
        if (response.status === 200 && response.data) {
          try {
            const data = JSON.parse(response.data);
            if (endpoint === '/api/config' && data.mcpServers) {
              log('   ✅ 发现MCP配置', 'green');
              Object.keys(data.mcpServers).forEach(server => {
                log(`      - ${server}: ${data.mcpServers[server].disabled ? '禁用' : '启用'}`, 'blue');
              });
            }
          } catch (e) {
            // 不是JSON响应，跳过
          }
        }
      } catch (e) {
        log(`   ${endpoint}: 无法访问`, 'red');
      }
    }
  } catch (error) {
    log('❌ API检查失败: ' + error.message, 'red');
  }
}

async function checkLobeChatHTML() {
  log('\n🔍 检查LobeChat页面内容...', 'blue');
  
  try {
    const response = await makeRequest('http://localhost:3210');
    
    if (response.status === 200 || response.status === 307) {
      log('✅ LobeChat页面可访问', 'green');
      
      // 检查HTML内容中是否包含MCP相关信息
      const html = response.data;
      const mcpKeywords = [
        'mcp',
        'MCP',
        'Model Context Protocol',
        'graphiti',
        'plugin',
        'extension'
      ];
      
      let foundKeywords = [];
      mcpKeywords.forEach(keyword => {
        if (html.includes(keyword)) {
          foundKeywords.push(keyword);
        }
      });
      
      if (foundKeywords.length > 0) {
        log(`   ✅ 页面包含MCP相关内容: ${foundKeywords.join(', ')}`, 'green');
      } else {
        log('   ⚠️  页面未发现明显的MCP相关内容', 'yellow');
      }
      
      // 检查是否有JavaScript配置
      if (html.includes('mcpServers') || html.includes('MCP_ENABLED')) {
        log('   ✅ 发现MCP JavaScript配置', 'green');
      }
      
    } else {
      log(`❌ LobeChat页面访问失败: ${response.status}`, 'red');
    }
  } catch (error) {
    log('❌ 页面检查失败: ' + error.message, 'red');
  }
}

async function checkLobeChatSettings() {
  log('\n🔍 检查LobeChat设置存储...', 'blue');
  
  try {
    // 检查可能的设置存储位置
    const fs = require('fs');
    const path = require('path');
    
    const settingsPaths = [
      './data/lobechat/settings.json',
      './data/lobechat/config.json',
      './data/lobechat/plugins.json',
      './data/lobechat/.lobechat/settings.json'
    ];
    
    let foundSettings = false;
    
    for (const settingsPath of settingsPaths) {
      if (fs.existsSync(settingsPath)) {
        try {
          const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
          log(`   ✅ 找到设置文件: ${settingsPath}`, 'green');
          
          // 检查MCP相关设置
          if (settings.mcpServers) {
            log('   ✅ 包含MCP服务器配置', 'green');
            Object.keys(settings.mcpServers).forEach(server => {
              const config = settings.mcpServers[server];
              log(`      - ${server}: ${config.disabled ? '禁用' : '启用'}`, 'blue');
            });
            foundSettings = true;
          }
          
          if (settings.plugins) {
            log('   ✅ 包含插件配置', 'green');
            foundSettings = true;
          }
          
        } catch (e) {
          log(`   ❌ 设置文件格式错误: ${settingsPath}`, 'red');
        }
      }
    }
    
    if (!foundSettings) {
      log('   ⚠️  未找到MCP相关设置', 'yellow');
    }
    
  } catch (error) {
    log('❌ 设置检查失败: ' + error.message, 'red');
  }
}

function generateUIGuide() {
  log('\n📋 LobeChat界面检查指南', 'bold');
  log('='.repeat(50), 'bold');
  
  log('\n🔍 在LobeChat界面中查看MCP插件状态:', 'blue');
  log('1. 访问 http://localhost:3210', 'blue');
  log('2. 点击左下角的设置图标 ⚙️', 'blue');
  log('3. 查看以下位置:', 'blue');
  log('   - "插件设置" 或 "Extensions" 选项卡', 'blue');
  log('   - "MCP服务器" 或 "MCP Servers" 选项卡', 'blue');
  log('   - "高级设置" 中的MCP相关选项', 'blue');
  
  log('\n✅ 成功配置的标志:', 'green');
  log('- 在插件列表中看到 "Graphiti Memory" 或 "graphiti-memory"', 'green');
  log('- 插件状态显示为 "已启用" 或 "Enabled"', 'green');
  log('- MCP服务器列表中显示连接状态为 "已连接"', 'green');
  log('- 在对话中可以看到工具调用图标或提示', 'green');
  
  log('\n❌ 需要修复的标志:', 'red');
  log('- 插件列表为空或没有MCP相关插件', 'red');
  log('- 插件状态显示为 "已禁用" 或 "连接失败"', 'red');
  log('- 设置中没有MCP或插件相关选项', 'red');
  
  log('\n🔧 如果界面中看不到MCP插件:', 'yellow');
  log('1. 运行: node auto-configure-lobechat.js', 'yellow');
  log('2. 重启: docker-compose restart lobechat', 'yellow');
  log('3. 清除浏览器缓存并刷新页面', 'yellow');
  log('4. 检查浏览器控制台是否有错误信息', 'yellow');
}

async function main() {
  log('🚀 检查LobeChat界面中的MCP插件状态...', 'bold');
  
  await checkLobeChatHTML();
  await checkLobeChatAPI();
  await checkLobeChatSettings();
  
  generateUIGuide();
}

if (require.main === module) {
  main().catch(error => {
    log('❌ 检查过程中发生错误: ' + error.message, 'red');
    process.exit(1);
  });
}

module.exports = { main };