#!/usr/bin/env node
/**
 * LobeChat MCP自动配置脚本
 * 自动在LobeChat中配置Graphiti MCP插件
 */

const fs = require('fs');
const path = require('path');

// LobeChat数据目录
const LOBECHAT_DATA_DIR = './data/lobechat';
const CONFIG_FILE = path.join(LOBECHAT_DATA_DIR, 'config.json');

// MCP服务器配置
const MCP_CONFIG = {
  mcpServers: {
    "graphiti-memory": {
      name: "Graphiti Memory",
      url: "http://graphiti-mcp:8000/mcp/stream",
      transport: "http",
      disabled: false,
      autoApprove: [
        "add_episode",
        "search", 
        "get_episodes"
      ]
    }
  },
  settings: {
    enableMCP: true,
    mcpAutoConnect: true
  }
};

function ensureDirectory(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`✅ 创建目录: ${dir}`);
  }
}

function writeConfig() {
  try {
    ensureDirectory(LOBECHAT_DATA_DIR);
    
    let existingConfig = {};
    if (fs.existsSync(CONFIG_FILE)) {
      try {
        existingConfig = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
        console.log('📖 读取现有配置文件');
      } catch (e) {
        console.log('⚠️  现有配置文件格式错误，将创建新配置');
      }
    }
    
    // 合并配置
    const mergedConfig = {
      ...existingConfig,
      ...MCP_CONFIG,
      mcpServers: {
        ...existingConfig.mcpServers,
        ...MCP_CONFIG.mcpServers
      }
    };
    
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(mergedConfig, null, 2));
    console.log('✅ MCP配置已写入:', CONFIG_FILE);
    console.log('🔧 配置内容:', JSON.stringify(MCP_CONFIG, null, 2));
    
    return true;
  } catch (error) {
    console.error('❌ 配置写入失败:', error.message);
    return false;
  }
}

function main() {
  console.log('🚀 开始配置LobeChat MCP插件...');
  
  if (writeConfig()) {
    console.log('\n🎉 配置完成！');
    console.log('📋 下一步:');
    console.log('1. 重启LobeChat容器: docker-compose restart lobechat');
    console.log('2. 运行验证检查: node verify-mcp-config.js');
    console.log('3. 访问 http://localhost:3210');
    console.log('4. MCP插件应该已经自动配置好了');
  } else {
    console.log('\n❌ 配置失败，请手动配置MCP插件');
    console.log('📋 手动配置信息:');
    console.log('- 插件标识符: graphiti-memory');
    console.log('- 服务器URL: http://graphiti-mcp:8000/mcp/stream');
    console.log('- 传输方式: HTTP');
  }
}

if (require.main === module) {
  main();
}

module.exports = { writeConfig, MCP_CONFIG };