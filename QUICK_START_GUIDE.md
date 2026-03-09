# Antigravity 团队快速上手指南

## 🚀 5分钟快速启动

### 第一步：环境准备
```bash
# 1. 进入项目目录
cd project_code/graphiti-demo

# 2. 复制环境配置
cp .env.example .env

# 3. 编辑环境变量（重要！）
# 在 .env 文件中配置你的老张API密钥
OPENAI_API_KEY=your_laozhang_api_key_here
```

### 第二步：启动服务
```bash
# 启动所有服务
docker-compose up -d

# 等待30秒让服务完全启动
sleep 30

# 验证服务状态
docker-compose ps
```

### 第三步：验证功能
```bash
# 快速功能测试
python test-final-memory.py
```

**如果看到 "🎉 记忆功能完全修复，可以正常使用！"，说明系统正常运行！**

## 🔧 常用命令速查

### 服务管理
```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f graphiti-mcp

# 查看服务状态
docker-compose ps
```

### 功能测试
```bash
# 综合功能测试
python test-final-memory.py

# 记忆功能测试
python test-memory-function.py

# 添加用户信息
python add-user-info.py

# 搜索功能调试
python debug-search.py
```

### API测试
```bash
# 健康检查
curl http://localhost:8000/health

# 获取工具列表
curl http://localhost:8000/tools/list

# 添加记忆
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "add_episode",
    "arguments": {
      "name": "快速测试",
      "episode_body": "这是一个快速测试记忆"
    }
  }'
```

## 🐛 常见问题解决

### 问题1：服务启动失败
```bash
# 检查端口占用
netstat -an | grep 8000
netstat -an | grep 6379

# 如果端口被占用，修改docker-compose.yml中的端口映射
```

### 问题2：API调用失败
```bash
# 检查API密钥配置
grep OPENAI_API_KEY .env

# 测试API连接
curl -H "Authorization: Bearer your_api_key" \
  https://api.laozhang.ai/v1/models
```

### 问题3：搜索无结果
```bash
# 检查数据库连接
docker-compose logs falkordb

# 重新添加测试数据
python add-user-info.py
```

### 问题4：LobeChat连接失败
1. 确保MCP服务器运行在 `http://localhost:8000`
2. 在LobeChat中配置MCP插件
3. 检查网络连接和防火墙设置

## 📁 重要文件说明

### 核心文件
- `mcp_server/graphiti_mcp_server.py` - 主服务器代码
- `docker-compose.yml` - 服务编排配置
- `.env` - 环境变量配置
- `requirements.txt` - Python依赖

### 测试文件
- `test-final-memory.py` - 综合功能测试
- `test-memory-function.py` - 记忆功能测试
- `add-user-info.py` - 用户信息添加
- `debug-search.py` - 搜索功能调试

### 文档文件
- `HANDOVER_TO_ANTIGRAVITY.md` - 详细移交文档
- `DEVELOPMENT_STANDARDS.md` - 开发规范
- `ANTIGRAVITY_CHECKLIST.md` - 接手检查清单
- `MEMORY_FUNCTION_SUCCESS.md` - 功能修复报告

## 🎯 开发工作流程

### 1. 日常开发
```bash
# 1. 拉取最新代码
git pull origin main

# 2. 创建功能分支
git checkout -b feature/new-feature

# 3. 开发和测试
# 编辑代码...
python test-final-memory.py

# 4. 代码质量检查
black mcp_server/
flake8 mcp_server/

# 5. 提交代码
git add .
git commit -m "feat: 添加新功能"
git push origin feature/new-feature
```

### 2. 部署流程
```bash
# 1. 合并到主分支
git checkout main
git merge feature/new-feature

# 2. 重启服务
docker-compose down
docker-compose up -d

# 3. 验证部署
python test-final-memory.py
```

## 📊 监控和维护

### 日常监控
```bash
# 查看服务状态
docker-compose ps

# 查看资源使用
docker stats

# 查看日志
docker-compose logs --tail=100 graphiti-mcp

# 检查磁盘空间
df -h
```

### 数据备份
```bash
# 备份FalkorDB数据
docker exec graphiti-demo-falkordb redis-cli BGSAVE

# 备份配置文件
cp .env .env.backup.$(date +%Y%m%d)
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d)
```

## 🔒 安全检查

### 定期检查项目
- [ ] API密钥是否安全存储
- [ ] 日志中是否有敏感信息泄露
- [ ] 服务是否只监听必要端口
- [ ] 数据备份是否定期执行

## 📞 获得帮助

### 内部资源
1. **详细文档**: 阅读 `HANDOVER_TO_ANTIGRAVITY.md`
2. **开发规范**: 参考 `DEVELOPMENT_STANDARDS.md`
3. **问题排查**: 使用 `ANTIGRAVITY_CHECKLIST.md`

### 外部资源
1. **Graphiti文档**: https://github.com/getzep/graphiti
2. **MCP协议**: https://modelcontextprotocol.io/
3. **FastAPI文档**: https://fastapi.tiangolo.com/

### 紧急联系
- **技术问题**: 查看项目日志和错误信息
- **API问题**: 检查老张API服务状态
- **部署问题**: 验证Docker和网络配置

---

**提示**: 这个指南涵盖了90%的日常操作。如需更详细信息，请参考完整的移交文档。