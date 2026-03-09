#!/bin/bash

# Graphiti演示项目启动脚本

echo "🚀 启动Graphiti演示项目..."

# 检查.env文件
if [ ! -f .env ]; then
    echo "⚠️  未找到.env文件，从示例创建..."
    cp .env.example .env
    echo "📝 请编辑.env文件，添加你的OPENAI_API_KEY"
    echo "   然后重新运行此脚本"
    exit 1
fi

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 未找到Docker，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ 未找到docker-compose，请先安装docker-compose"
    exit 1
fi

# 创建必要的目录
mkdir -p data logs

# 启动服务
echo "🐳 启动Docker服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

echo ""
echo "✅ 启动完成！"
echo ""
echo "📱 访问地址："
echo "   Web界面: http://localhost:3000"
echo "   MCP服务器: http://localhost:8000"
echo ""
echo "🛠️  管理命令："
echo "   查看日志: docker-compose logs -f"
echo "   停止服务: docker-compose down"
echo "   重启服务: docker-compose restart"
echo ""