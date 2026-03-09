# LobeChat 设置指南

## 🚀 快速开始

### 1. 访问 LobeChat
打开浏览器访问：http://localhost:3210

### 2. 配置 API 密钥

#### 方法一：在设置中配置（推荐）
1. 点击右上角的设置图标 ⚙️
2. 选择 "语言模型" 或 "Language Models"
3. 找到 "OpenAI" 配置项
4. 输入以下信息：
   - **API Key**: `sk-rmMS3NM1iiJI7BkzF153946dCaA4491a9cD73907F7001834`
   - **API Endpoint**: `https://api.laozhang.ai/v1`
   - **Model**: `gpt-3.5-turbo`

#### 方法二：通过环境变量（已配置）
环境变量已经设置，但可能需要在界面中激活：
- `OPENAI_API_KEY`: 已设置
- `OPENAI_PROXY_URL`: 已设置为老张API
- `API_KEY_SELECT_MODE`: 设置为 manual

### 3. 配置 MCP 插件（记忆功能）

1. 在设置中找到 "插件" 或 "Plugins" 选项
2. 添加新的 MCP 插件：
   - **插件名称**: `graphiti-memory`
   - **服务器URL**: `http://graphiti-mcp:8000/mcp/stream`
   - **传输方式**: `http`
3. 启用插件

### 4. 开始对话

现在你可以：
1. 创建新的对话
2. AI 助手将具有长期记忆功能
3. 你的对话内容会被存储在 Graphiti 知识图谱中
4. AI 可以记住之前的对话内容并在后续对话中引用

## 🔧 故障排除

### "Failed to fetch" 错误
如果遇到此错误，请检查：

1. **API 密钥配置**：
   - 确保在 LobeChat 设置中正确配置了 API 密钥
   - 检查 API 端点是否正确设置为 `https://api.laozhang.ai/v1`

2. **网络连接**：
   - 确保可以访问老张API
   - 检查防火墙设置

3. **服务状态**：
   ```bash
   # 检查所有服务状态
   docker-compose ps
   
   # 查看 LobeChat 日志
   docker-compose logs lobechat
   
   # 查看 MCP 服务器日志
   docker-compose logs graphiti-mcp
   ```

### MCP 插件无法连接
如果 MCP 插件无法工作：

1. 检查 MCP 服务器状态：
   ```bash
   curl http://localhost:8000/health
   ```

2. 检查 MCP 服务器日志：
   ```bash
   docker-compose logs graphiti-mcp
   ```

3. 验证插件配置：
   - URL: `http://graphiti-mcp:8000/mcp/stream`
   - 传输方式: `http`

## 🎯 使用示例

### 基本对话
```
用户: 你好，我是李林松，一名软件工程师。
AI: 你好李林松！很高兴认识你。作为一名软件工程师，你主要专注于哪些技术领域呢？

用户: 我主要做后端开发，熟悉 Python 和 Java。
AI: 很棒！Python 和 Java 都是后端开发的热门选择。你有什么特别的项目经验想分享吗？
```

### 记忆功能测试
```
用户: 请记住，我最喜欢的编程语言是 Python。
AI: 好的，我已经记住了你最喜欢的编程语言是 Python。

（在新的对话中）
用户: 你还记得我最喜欢什么编程语言吗？
AI: 当然记得！你最喜欢的编程语言是 Python。
```

## 📊 功能特性

### ✅ 已启用功能
- 🤖 AI 对话（使用老张API）
- 🧠 长期记忆（Graphiti 知识图谱）
- 🔌 MCP 插件支持
- 💾 对话历史保存
- 🎨 现代化界面

### 🔧 技术栈
- **前端**: LobeChat (Next.js)
- **AI API**: 老张API (OpenAI 兼容)
- **记忆存储**: Graphiti + FalkorDB
- **协议**: MCP (Model Context Protocol)

## 🆘 获取帮助

如果遇到问题：
1. 查看本指南的故障排除部分
2. 检查 Docker 容器日志
3. 确认所有服务都在运行
4. 验证网络连接

---

🎉 **现在你可以享受具有长期记忆功能的 AI 助手了！**