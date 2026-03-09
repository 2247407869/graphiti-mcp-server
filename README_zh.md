# Graphiti演示项目 - 老张API版本

基于Graphiti知识图谱的MCP协议演示项目，使用老张API提供LLM和嵌入服务。

## 🎯 项目特色

- ✅ **老张API直连**：原生支持OpenAI Responses API，无需代理转换
- ✅ **知识图谱记忆**：基于Graphiti的智能记忆管理
- ✅ **MCP协议支持**：标准化的工具调用接口
- ✅ **LobeChat集成**：专业的聊天交互界面
- ✅ **Docker一键部署**：完整的容器化解决方案

## 🚀 快速开始

### 1. 环境准备

确保已安装：
- Docker & Docker Compose
- 老张API密钥

### 2. 配置API密钥

编辑 `.env` 文件：
```bash
# 老张API配置
OPENAI_API_KEY=your-laozhang-api-key
OPENAI_BASE_URL=https://api.laozhang.ai/v1
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
```

### 3. 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

### 4. 访问服务

- **LobeChat界面**: http://localhost:3210
- **Web演示界面**: http://localhost:3000  
- **MCP服务器**: http://localhost:8000
- **FalkorDB**: localhost:6379

## 🔧 服务架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LobeChat      │    │   Web Demo       │    │   老张API       │
│   (Port 3210)   │    │   (Port 3000)    │    │   (直连)        │
└─────────┬───────┘    └─────────┬────────┘    └─────────────────┘
          │                      │                       ▲
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Graphiti MCP Server   │
                    │      (Port 8000)        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      FalkorDB           │
                    │      (Port 6379)        │
                    └─────────────────────────┘
```

## 🛠️ MCP工具使用

### 在LobeChat中使用

1. 访问 http://localhost:3210
2. 在聊天中使用MCP工具：
   - `add_episode`: 添加记忆片段
   - `search`: 搜索记忆内容
   - `get_episodes`: 获取记忆列表

### 示例对话

```
用户: 请帮我记住：今天学习了Graphiti知识图谱技术
助手: [调用add_episode工具]
✅ 已成功添加记忆片段

用户: 我之前学过什么技术？
助手: [调用search工具]
根据记忆搜索，您学习过：Graphiti知识图谱技术...
```

## 📊 健康检查

```bash
# 检查MCP服务器状态
curl http://localhost:8000/health

# 检查可用工具
curl http://localhost:8000/tools/list

# 测试添加记忆
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "add_episode",
    "arguments": {
      "name": "测试记忆",
      "episode_body": "这是一个测试记忆片段"
    }
  }'
```

## 🔍 故障排除

### 常见问题

1. **API连接失败**
   - 检查老张API密钥是否正确
   - 确认网络连接正常

2. **容器启动失败**
   - 检查端口是否被占用
   - 查看容器日志：`docker-compose logs [service-name]`

3. **记忆功能异常**
   - 确认FalkorDB正常运行
   - 检查MCP服务器日志

### 查看日志

```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs graphiti-mcp
docker-compose logs lobechat
docker-compose logs falkordb
```

## 🎉 技术亮点

### 老张API优势
- ✅ **原生Responses API支持**：无需格式转换
- ✅ **完全OpenAI兼容**：标准接口调用
- ✅ **稳定可靠**：100%成功率验证
- ✅ **直连架构**：简化系统复杂度

### Graphiti知识图谱
- 🧠 **智能记忆管理**：自动构建知识关联
- 🔍 **语义搜索**：基于向量相似度的智能检索  
- 📊 **图谱可视化**：知识关系清晰展现
- 🔄 **动态更新**：实时学习和记忆更新

### MCP协议集成
- 🔧 **标准化接口**：统一的工具调用规范
- 🔌 **插件化架构**：易于扩展和集成
- 🚀 **高性能**：优化的通信协议
- 🛡️ **类型安全**：完整的参数验证

## 📝 更新日志

### v1.0.0 (2026-01-04)
- ✅ 成功集成老张API
- ✅ 实现Graphiti知识图谱功能
- ✅ 完成MCP协议支持
- ✅ 集成LobeChat界面
- ✅ 优化Docker部署方案
- ✅ 移除API代理，简化架构

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

MIT License

---

**项目状态**: 🟢 生产就绪  
**最后更新**: 2026-01-04  
**API提供商**: 老张API (直连模式)