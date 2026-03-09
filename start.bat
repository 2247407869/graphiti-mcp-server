@echo off
chcp 65001 >nul

echo 🚀 启动Graphiti演示项目...

REM 检查.env文件
if not exist .env (
    echo ⚠️  未找到.env文件，从示例创建...
    copy .env.example .env >nul
    echo 📝 请编辑.env文件，添加你的OPENAI_API_KEY
    echo    然后重新运行此脚本
    pause
    exit /b 1
)

REM 检查Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Docker，请先安装Docker Desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到docker-compose，请先安装docker-compose
    pause
    exit /b 1
)

REM 创建必要的目录
if not exist data mkdir data
if not exist logs mkdir logs

REM 启动服务
echo 🐳 启动Docker服务...
docker-compose up -d

REM 等待服务启动
echo ⏳ 等待服务启动...
timeout /t 10 /nobreak >nul

REM 检查服务状态
echo 🔍 检查服务状态...
docker-compose ps

echo.
echo ✅ 启动完成！
echo.
echo 📱 访问地址：
echo    Web界面: http://localhost:3000
echo    MCP服务器: http://localhost:8000
echo.
echo 🛠️  管理命令：
echo    查看日志: docker-compose logs -f
echo    停止服务: docker-compose down
echo    重启服务: docker-compose restart
echo.
pause