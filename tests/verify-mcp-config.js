#!/usr/bin/env node
/**
 * LobeChat MCP配置验证脚本
 * 检查MCP插件是否正确配置并可用
 */

const fs = require('fs');
const path = require('path');
const http = require('http');

// 配置
const LOBECHAT_DATA_DIR = './data/lobechat';
const CONFIG_FILE = path.join(LOBECHAT_DATA_DIR, 'config.json');
const MCP_SERVER_URL = 'http://localhost:8000';
const LOBECHAT_URL = 'http://localhost:3210';

// 颜色输出
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

async function checkConfigFile() {
  log('\n🔍 检查LobeChat配置文件...', 'blue');
  
  try {
    if (!fs.existsSync(CONFIG_FILE)) {
      log('❌ 配置文件不存在: ' + CONFIG_FILE, 'red');
      return false;
    }

    const config = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
    
    if (!config.mcpServers) {
      log('❌ 配置文件中没有mcpServers配置', 'red');
      return false;
    }

    if (!config.mcpServers['graphiti-memory']) {
      log('❌ 配置文件中没有graphiti-memory服务器配置', 'red');
      return false;
    }

    const mcpConfig = config.mcpServers['graphiti-memory'];
    log('✅ 配置文件存在并包含MCP配置', 'green');
    log(`   服务器名称: ${mcpConfig.name || 'graphiti-memory'}`, 'blue');
    log(`   服务器URL: ${mcpConfig.url}`, 'blue');
    log(`   传输方式: ${mcpConfig.transport}`, 'blue');
    log(`   是否禁用: ${mcpConfig.disabled}`, 'blue');
    log(`   自动批准工具: ${mcpConfig.autoApprove?.join(', ') || '无'}`, 'blue');

    return true;
  } catch (error) {
    log('❌ 读取配置文件失败: ' + error.message, 'red');
    return false;
  }
}

async function checkMCPServer() {
  log('\n🔍 检查MCP服务器状态...', 'blue');
  
  try {
    // 检查健康状态
    const healthResponse = await makeRequest(`${MCP_SERVER_URL}/health`);
    if (healthResponse.status !== 200) {
      log('❌ MCP服务器健康检查失败', 'red');
      return false;
    }

    log('✅ MCP服务器运行正常', 'green');
    log(`   状态: ${healthResponse.data.status}`, 'blue');
    log(`   模式: ${healthResponse.data.mode}`, 'blue');
    log(`   Graphiti可用: ${healthResponse.data.graphiti_available}`, 'blue');

    // 检查MCP流式端点
    const mcpResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
      method: 'POST',
      body: {
        jsonrpc: '2.0',
        id: 'test',
        method: 'initialize',
        params: {}
      }
    });

    if (mcpResponse.status !== 200) {
      log('❌ MCP流式端点不可用', 'red');
      return false;
    }

    log('✅ MCP流式端点工作正常', 'green');
    log(`   协议版本: ${mcpResponse.data.result?.protocolVersion}`, 'blue');

    // 检查工具列表
    const toolsResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
      method: 'POST',
      body: {
        jsonrpc: '2.0',
        id: 'test2',
        method: 'tools/list',
        params: {}
      }
    });

    if (toolsResponse.status === 200 && toolsResponse.data.result?.tools) {
      log('✅ MCP工具列表可用', 'green');
      toolsResponse.data.result.tools.forEach(tool => {
        log(`   - ${tool.name}: ${tool.description}`, 'blue');
      });
    }

    return true;
  } catch (error) {
    log('❌ MCP服务器检查失败: ' + error.message, 'red');
    return false;
  }
}

async function checkLobeChatAccess() {
  log('\n🔍 检查LobeChat访问...', 'blue');
  
  try {
    const response = await makeRequest(LOBECHAT_URL);
    // LobeChat可能返回重定向状态码，这是正常的
    if (response.status === 200 || response.status === 307 || response.status === 302) {
      log('✅ LobeChat可以访问', 'green');
      return true;
    } else {
      log(`❌ LobeChat访问失败，状态码: ${response.status}`, 'red');
      return false;
    }
  } catch (error) {
    log('❌ LobeChat访问失败: ' + error.message, 'red');
    return false;
  }
}

async function checkDockerServices() {
  log('\n🔍 检查Docker服务状态...', 'blue');
  
  try {
    const { exec } = require('child_process');
    
    return new Promise((resolve) => {
      exec('docker-compose ps --format json', { cwd: __dirname }, (error, stdout, stderr) => {
        if (error) {
          log('❌ 无法检查Docker服务状态: ' + error.message, 'red');
          resolve(false);
          return;
        }

        try {
          const services = stdout.trim().split('\n')
            .filter(line => line.trim())
            .map(line => JSON.parse(line));

          const requiredServices = ['graphiti-demo-mcp', 'graphiti-demo-lobechat', 'graphiti-demo-falkordb'];
          let allRunning = true;

          requiredServices.forEach(serviceName => {
            const service = services.find(s => s.Name === serviceName);
            if (service) {
              const isRunning = service.State === 'running';
              log(`${isRunning ? '✅' : '❌'} ${serviceName}: ${service.State}`, isRunning ? 'green' : 'red');
              if (!isRunning) allRunning = false;
            } else {
              log(`❌ ${serviceName}: 未找到`, 'red');
              allRunning = false;
            }
          });

          resolve(allRunning);
        } catch (parseError) {
          log('❌ 解析Docker服务状态失败', 'red');
          resolve(false);
        }
      });
    });
  } catch (error) {
    log('❌ Docker服务检查失败: ' + error.message, 'red');
    return false;
  }
}

async function testMCPIntegration() {
  log('\n🔍 测试MCP集成功能...', 'blue');
  
  try {
    // 测试添加Episode
    const addResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
      method: 'POST',
      body: {
        jsonrpc: '2.0',
        id: 'test3',
        method: 'tools/call',
        params: {
          name: 'add_episode',
          arguments: {
            name: '配置验证测试',
            episode_body: '这是一个用于验证MCP配置的测试记忆片段'
          }
        }
      }
    });

    // 检查新的响应格式
    if (addResponse.status === 200 && addResponse.data.result?.content) {
      const content = addResponse.data.result.content[0];
      if (content && content.text) {
        const result = JSON.parse(content.text);
        if (result.success) {
          log('✅ MCP工具调用测试成功', 'green');
          log(`   Episode ID: ${result.episode_id}`, 'blue');
          
          // 测试搜索
          const searchResponse = await makeRequest(`${MCP_SERVER_URL}/mcp/stream`, {
            method: 'POST',
            body: {
              jsonrpc: '2.0',
              id: 'test4',
              method: 'tools/call',
              params: {
                name: 'search',
                arguments: {
                  query: '配置验证',
                  num_results: 1
                }
              }
            }
          });

          if (searchResponse.status === 200 && searchResponse.data.result?.content) {
            const searchContent = searchResponse.data.result.content[0];
            if (searchContent && searchContent.text) {
              const searchResults = JSON.parse(searchContent.text);
              if (Array.isArray(searchResults)) {
                log('✅ MCP搜索功能测试成功', 'green');
                log(`   找到 ${searchResults.length} 个结果`, 'blue');
                return true;
              }
            }
          }
        }
      }
    }

    log('❌ MCP集成功能测试失败', 'red');
    return false;
  } catch (error) {
    log('❌ MCP集成测试失败: ' + error.message, 'red');
    return false;
  }
}

function generateReport(results) {
  log('\n' + '='.repeat(60), 'bold');
  log('📊 MCP配置验证报告', 'bold');
  log('='.repeat(60), 'bold');

  const checks = [
    { name: 'Docker服务状态', result: results.docker },
    { name: 'MCP服务器状态', result: results.mcpServer },
    { name: 'LobeChat访问', result: results.lobechat },
    { name: '配置文件检查', result: results.configFile },
    { name: 'MCP集成功能', result: results.integration }
  ];

  let passedCount = 0;
  checks.forEach(check => {
    const status = check.result ? '✅ 通过' : '❌ 失败';
    const color = check.result ? 'green' : 'red';
    log(`${check.name}: ${status}`, color);
    if (check.result) passedCount++;
  });

  log('\n' + '='.repeat(60), 'bold');
  log(`总体状态: ${passedCount}/${checks.length} 项检查通过`, passedCount === checks.length ? 'green' : 'yellow');

  if (passedCount === checks.length) {
    log('\n🎉 所有检查通过！MCP配置完全正常', 'green');
    log('\n📋 下一步操作:', 'blue');
    log('1. 访问 LobeChat: http://localhost:3210', 'blue');
    log('2. 配置DeepSeek API密钥', 'blue');
    log('3. 开始与具有记忆功能的AI助手对话！', 'blue');
  } else {
    log('\n⚠️  部分检查失败，需要修复问题', 'yellow');
    log('\n🔧 建议操作:', 'blue');
    
    if (!results.docker) {
      log('- 启动Docker服务: docker-compose up -d', 'blue');
    }
    if (!results.configFile) {
      log('- 运行配置脚本: node auto-configure-lobechat.js', 'blue');
    }
    if (!results.lobechat) {
      log('- 重启LobeChat: docker-compose restart lobechat', 'blue');
    }
  }
}

async function main() {
  log('🚀 开始验证LobeChat MCP配置...', 'bold');

  const results = {
    docker: await checkDockerServices(),
    mcpServer: await checkMCPServer(),
    lobechat: await checkLobeChatAccess(),
    configFile: await checkConfigFile(),
    integration: false
  };

  // 只有在前面的检查都通过时才测试集成功能
  if (results.docker && results.mcpServer) {
    results.integration = await testMCPIntegration();
  }

  generateReport(results);
}

if (require.main === module) {
  main().catch(error => {
    log('❌ 验证过程中发生错误: ' + error.message, 'red');
    process.exit(1);
  });
}

module.exports = { main };