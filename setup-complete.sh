#!/bin/bash
# Graphiti演示项目一键完整设置脚本

echo "🚀 开始Graphiti演示项目完整设置..."

# 1. 启动所有服务
echo "📦 启动Docker服务..."
docker-compose up -d

# 2. 等待服务启动
echo "⏳ 等待服务启动完成..."
sleep 15

# 3. 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

# 4. 自动配置LobeChat MCP插件
echo "🔧 自动配置LobeChat MCP插件..."
node auto-configure-lobechat.js

# 5. 重启LobeChat使配置生效
echo "🔄 重启LobeChat使配置生效..."
docker-compose restart lobechat

# 6. 等待LobeChat重启
echo "⏳ 等待LobeChat重启完成..."
sleep 10

# 7. 运行配置验证
echo "🔍 验证MCP配置..."
node verify-mcp-config.js

echo ""
echo "🎉 设置完成！"
echo ""
echo "📋 访问地址:"
echo "- LobeChat聊天界面: http://localhost:3210"
echo "- 记忆管理界面: http://localhost:3000"
echo "- MCP API服务: http://localhost:8000"
echo "- 使用指南: http://localhost:3000/mcp-usage-guide.html"
echo ""
echo "🔑 DeepSeek API配置:"
echo "- API端点: https://api.deepseek.com/v1"
echo "- API密钥: sk-8bd504b2c56e4d9dbb78fac111ac9565"
echo "- 模型: deepseek-chat"
echo ""
echo "🔧 验证工具:"
echo "- 快速检查: node quick-check.js"
echo "- 详细验证: node verify-mcp-config.js"
echo ""
echo "✅ MCP插件已自动配置并验证，直接开始使用即可！"