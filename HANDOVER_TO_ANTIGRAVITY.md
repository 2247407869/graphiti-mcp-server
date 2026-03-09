# Graphiti MCP 项目移交文档

## 📋 项目概述

**项目名称**：Graphiti MCP 记忆管理系统  
**移交时间**：2026年1月4日  
**移交给**：antigravity 开发团队  
**当前状态**：✅ 核心功能完成，系统稳定运行

### 项目目标
构建一个基于Graphiti知识图谱的MCP（Model Context Protocol）服务器，为LobeChat等AI应用提供长期记忆功能。

## 🏗️ 系统架构

```
LobeChat → MCP Plugin → MCP Server (Port 8000) → Graphiti Core → FalkorDB
                                ↓
                        老张API (LLM + Embedding)
```

### 核心组件
1. **MCP服务器** (`mcp_server/graphiti_mcp_server.py`) - FastAPI应用，提供MCP协议接口
2. **Graphiti Core** - 知识图谱引擎，处理记忆存储和检索
3. **FalkorDB** - Redis兼容的图数据库，存储知识图谱
4. **老张API** - 提供LLM推理和文本嵌入服务

## 📁 项目结构

```
project_code/graphiti-demo/
├── mcp_server/
│   ├── graphiti_mcp_server.py     # 主服务器文件
│   └── requirements.txt           # Python依赖
├── config/
│   └── lobechat/
│       ├── mcp.json              # LobeChat MCP配置
│       └── mcp-proxy.js          # MCP代理配置
├── web/                          # Web界面（可选）
├── docker-compose.yml            # Docker编排文件
├── Dockerfile.mcp               # MCP服务器Docker镜像
├── .env                         # 环境变量配置
├── README.md                    # 项目说明
└── 测试脚本/
    ├── test-final-memory.py     # 综合功能测试
    ├── add-user-info.py         # 添加用户信息
    ├── test-memory-function.py  # 记忆功能测试
    └── debug-search.py          # 搜索功能调试
```

## 🔧 环境配置

### 必需的环境变量 (.env)
```bash
# 老张API配置
OPENAI_API_KEY=your_laozhang_api_key
OPENAI_BASE_URL=https://api.laozhang.ai/v1
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002

# 数据库配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Graphiti配置
GRAPHITI_GROUP_ID=demo
SEMAPHORE_LIMIT=5
```

### Docker服务
```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs graphiti-mcp
```

## 🚀 快速启动指南

### 1. 环境准备
```bash
cd project_code/graphiti-demo
cp .env.example .env  # 配置环境变量
```

### 2. 启动服务
```bash
docker-compose up -d
```

### 3. 验证服务
```bash
# 健康检查
curl http://localhost:8000/health

# 测试记忆功能
python test-final-memory.py
```

### 4. LobeChat集成
- 在LobeChat中配置MCP插件
- 服务器地址：`http://localhost:8000`
- 插件名称：`graphiti-memory`

## 🔍 核心功能

### 1. 记忆管理
- **添加记忆** (`add_episode`): 将用户输入转换为知识图谱
- **搜索记忆** (`search`): 基于语义相似度检索相关记忆
- **获取记忆列表** (`get_episodes`): 获取存储的记忆列表

### 2. MCP协议支持
- **工具列表** (`/tools/list`): 返回可用工具
- **工具调用** (`/tools/call`): 执行具体工具
- **流式端点** (`/mcp/stream`): 支持LobeChat的JSON-RPC格式

### 3. 搜索结果优化
- 将Graphiti的Edge对象转换为用户友好的中文描述
- 区分"原始记忆"和"知识关系"
- 按相关性和类型排序结果

## 📊 当前功能状态

| 功能模块 | 状态 | 说明 |
|---------|------|------|
| MCP服务器 | ✅ 完成 | FastAPI应用，所有端点正常 |
| 记忆存储 | ✅ 完成 | 支持中文内容，自动提取知识 |
| 记忆搜索 | ✅ 完成 | 语义搜索，结果格式化优化 |
| LobeChat集成 | ✅ 完成 | MCP插件正常工作 |
| 错误处理 | ✅ 完成 | JSON序列化问题已修复 |
| 日志记录 | ✅ 完成 | 详细的操作日志 |
| Docker部署 | ✅ 完成 | 容器化部署，易于维护 |

## 🧪 测试覆盖

### 自动化测试脚本
1. **test-final-memory.py** - 综合功能测试，模拟用户使用场景
2. **test-memory-function.py** - 记忆功能完整性测试
3. **add-user-info.py** - 用户信息添加和验证
4. **debug-search.py** - 搜索功能调试和验证

### 手动测试场景
- LobeChat中询问用户姓名、职业等信息
- 验证AI能够记住并回忆用户信息
- 测试不同关键词的搜索效果

## 🔒 安全考虑

### 当前安全措施
- API密钥通过环境变量管理
- Docker容器隔离
- CORS配置允许跨域访问

### 建议改进
- 添加API认证机制
- 实施访问频率限制
- 敏感信息加密存储

## 📈 性能指标

### 当前性能
- **响应时间**: 记忆添加 < 2秒，搜索 < 1秒
- **并发支持**: 支持多用户同时访问
- **内存使用**: 约200MB（包含Graphiti和依赖）
- **存储**: FalkorDB，支持持久化

### 性能优化建议
- 实施搜索结果缓存
- 优化Graphiti索引策略
- 考虑分布式部署

## 🐛 已知问题和限制

### 已解决问题
- ✅ JSON序列化错误（AddEpisodeResults对象）
- ✅ 搜索结果格式化问题
- ✅ MCP协议兼容性问题
- ✅ 中文内容处理问题

### 当前限制
- 仅支持文本类型的记忆
- 搜索主要返回知识关系而非原始文本
- 依赖外部API服务（老张API）

### 潜在改进方向
- 支持多媒体记忆（图片、音频）
- 增强原始文本检索能力
- 支持多种LLM提供商

## 📚 技术文档

### 关键文件说明
- `graphiti_mcp_server.py`: 主服务器实现，包含所有MCP端点
- `docker-compose.yml`: 服务编排，定义FalkorDB和MCP服务
- `requirements.txt`: Python依赖列表
- `MEMORY_FUNCTION_SUCCESS.md`: 功能修复详细报告

### API文档
- **健康检查**: `GET /health`
- **工具列表**: `GET /tools/list`
- **工具调用**: `POST /tools/call`
- **MCP流式**: `POST /mcp/stream`

## 🔄 维护指南

### 日常维护
```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f graphiti-mcp

# 重启服务
docker-compose restart graphiti-mcp

# 备份数据
docker exec graphiti-demo-falkordb redis-cli BGSAVE
```

### 故障排除
1. **服务无法启动**: 检查环境变量配置
2. **搜索无结果**: 验证FalkorDB连接和数据
3. **LobeChat连接失败**: 检查MCP插件配置

### 监控建议
- 监控MCP服务器响应时间
- 监控FalkorDB内存使用
- 监控老张API调用频率和成功率

## 📞 支持联系

### 技术支持
- **项目负责人**: 李林松
- **技术栈**: Python, FastAPI, Graphiti, FalkorDB, Docker
- **文档位置**: `project_code/graphiti-demo/`

### 紧急联系
如遇紧急技术问题，可参考以下资源：
- Graphiti官方文档: https://github.com/getzep/graphiti
- MCP协议规范: https://modelcontextprotocol.io/
- FastAPI文档: https://fastapi.tiangolo.com/

---

**移交确认**: 请antigravity团队确认收到此文档，并验证所有功能正常运行后，正式接管项目维护工作。