# LobeChat中使用Graphiti记忆MCP的完整指南

## 🎯 概述

LobeChat支持MCP（Model Context Protocol），可以通过MCP服务器扩展AI助手的功能。我们的Graphiti MCP服务器提供了记忆管理功能。

## 📋 配置步骤

### 方法1: 通过LobeChat界面配置（推荐）

1. **访问LobeChat设置**
   - 打开 http://localhost:3210
   - 点击左下角的设置图标 ⚙️
   - 选择"模型服务商"或"Model Providers"

2. **配置OpenAI兼容服务**
   - 添加自定义OpenAI服务
   - API端点: `https://api.deepseek.com/v1`
   - API密钥: `sk-8bd504b2c56e4d9dbb78fac111ac9565`
   - 模型: `deepseek-chat`

3. **启用MCP功能**
   - 在设置中找到"插件"或"Plugins"选项
   - 启用MCP支持
   - 添加MCP服务器配置

4. **配置Graphiti MCP服务器**
   ```json
   {
     "name": "graphiti-memory",
     "url": "http://localhost:8000",
     "description": "Graphiti记忆管理服务",
     "tools": [
       "add_episode",
       "search", 
       "get_episodes"
     ]
   }
   ```

### 方法2: 直接使用HTTP API（当前可用）

由于我们的MCP服务器是HTTP接口，你可以直接在LobeChat中通过以下方式使用：

#### 🔍 搜索记忆
在聊天中输入：
```
请帮我搜索关于"项目进展"的记忆
```

然后AI助手可以调用我们的搜索API：
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "search",
    "arguments": {
      "query": "项目进展",
      "num_results": 5
    }
  }'
```

#### 💾 添加记忆
在聊天中说：
```
请帮我记住：今天讨论了新功能的设计方案
```

AI助手可以调用添加记忆API：
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "add_episode",
    "arguments": {
      "name": "功能设计讨论",
      "episode_body": "今天讨论了新功能的设计方案"
    }
  }'
```

## 🛠️ 手动配置MCP（高级用户）

如果你想手动配置MCP，可以按以下步骤：

### 1. 创建MCP配置文件

在LobeChat的配置目录中创建 `mcp.json`：

```json
{
  "mcpServers": {
    "graphiti": {
      "command": "curl",
      "args": [
        "-X", "POST",
        "http://localhost:8000/tools/call",
        "-H", "Content-Type: application/json",
        "-d", "@-"
      ],
      "env": {},
      "disabled": false,
      "autoApprove": [
        "add_episode",
        "search",
        "get_episodes"
      ]
    }
  }
}
```

### 2. 重启LobeChat服务

```bash
docker-compose restart lobechat
```

## 🎮 使用示例

### 基本对话流程

1. **开始对话**
   ```
   用户: 你好，我想讨论一个新项目的想法
   AI: 你好！我很乐意听听你的新项目想法。请告诉我更多细节。
   ```

2. **AI自动记忆**
   ```
   用户: 这个项目是关于开发一个智能家居控制系统
   AI: 很有趣的项目！让我记住这个信息...
   [AI调用add_episode API记录这次对话]
   ```

3. **后续对话中回忆**
   ```
   用户: 我们之前讨论过什么项目？
   AI: 让我搜索一下我们的对话记录...
   [AI调用search API查找相关记忆]
   AI: 我们之前讨论过开发一个智能家居控制系统的项目。
   ```

### 高级功能

#### 🔍 主动搜索相关记忆
```
用户: 关于智能家居，我们还讨论过什么？
AI: [搜索"智能家居"相关记忆并展示结果]
```

#### 📊 查看记忆统计
```
用户: 我们总共聊过多少话题？
AI: [调用get_episodes API获取统计信息]
```

#### 🎯 精确记忆管理
```
用户: 请记住：智能家居项目预算是10万元
AI: [将这个重要信息作为独立记忆保存]
```

## 🔧 故障排除

### 常见问题

1. **MCP服务器连接失败**
   - 检查MCP服务器状态: `curl http://localhost:8000/health`
   - 确认LobeChat可以访问MCP服务器

2. **工具调用失败**
   - 检查API格式是否正确
   - 查看LobeChat和MCP服务器的日志

3. **记忆功能不工作**
   - 确认Graphiti服务正常运行
   - 检查FalkorDB连接状态

### 调试命令

```bash
# 检查所有服务状态
docker-compose ps

# 查看LobeChat日志
docker-compose logs -f lobechat

# 查看MCP服务器日志  
docker-compose logs -f graphiti-mcp

# 测试MCP API
curl http://localhost:8000/tools/list
```

## 📚 更多资源

- [LobeChat MCP文档](https://lobehub.com/docs/usage/plugins/mcp)
- [MCP协议规范](https://modelcontextprotocol.io/)
- [Graphiti项目文档](https://github.com/getzep/graphiti)

## 💡 使用技巧

1. **明确指示**: 告诉AI你想要记住或搜索什么
2. **结构化信息**: 提供清晰的上下文信息
3. **定期回顾**: 让AI搜索和总结之前的对话
4. **分类管理**: 为不同类型的记忆使用不同的名称

现在你可以在LobeChat中享受具有记忆功能的AI助手了！