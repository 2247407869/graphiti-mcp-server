# Graphiti Embeddings 配置说明

## 问题说明

Graphiti需要embeddings API来进行语义搜索，但DeepSeek目前不提供embeddings端点。

## 解决方案

### 方案1：使用OpenAI API密钥（推荐）

1. 获取OpenAI API密钥
2. 在`.env`文件中添加：
   ```
   OPENAI_EMBEDDINGS_API_KEY=sk-your-openai-key-here
   ```
3. 重启服务：
   ```bash
   docker-compose restart graphiti-mcp
   ```

### 方案2：使用其他embeddings提供商

可以配置使用以下服务：
- Hugging Face Inference API
- Cohere API
- Azure OpenAI
- 本地embeddings模型

### 方案3：禁用embeddings功能

如果不需要语义搜索，可以修改代码使用简单的文本匹配。

## 当前状态

- ✅ Graphiti已连接到FalkorDB
- ❌ 缺少embeddings API配置
- ⚠️ 搜索功能将失败直到配置embeddings

## 配置检查

运行以下命令检查配置状态：
```bash
curl http://localhost:8000/health
```

查看 `embeddings_configured` 字段。

## 获取OpenAI API密钥

1. 访问 https://platform.openai.com/api-keys
2. 创建新的API密钥
3. 复制密钥到`.env`文件中

## 成本估算

OpenAI embeddings API成本很低：
- text-embedding-3-small: $0.00002 per 1K tokens
- 对于一般使用，每月成本通常不超过几美元