# Graphiti MCP 项目开发规范

## 📋 总则

本文档为antigravity团队接手Graphiti MCP项目后的开发规范，确保代码质量、系统稳定性和团队协作效率。

## 🏗️ 代码规范

### Python代码规范

#### 1. 代码风格
- **遵循PEP 8**：使用标准Python代码风格
- **行长度**：最大88字符（Black格式化器标准）
- **缩进**：使用4个空格，不使用Tab
- **编码**：所有文件使用UTF-8编码

#### 2. 命名规范
```python
# 类名：大驼峰命名
class GraphitiWrapper:
    pass

# 函数名：小写+下划线
async def search_episodes(query: str) -> List[Dict[str, Any]]:
    pass

# 变量名：小写+下划线
episode_body = "用户输入内容"
formatted_results = []

# 常量：全大写+下划线
REDIS_HOST = "localhost"
DEFAULT_TIMEOUT = 30
```

#### 3. 类型注解
```python
# 必须使用类型注解
from typing import Dict, List, Optional, Any

async def add_episode(
    self, 
    name: str, 
    episode_body: str, 
    episode_type: str = "text",
    source_description: str = "User input"
) -> Dict[str, Any]:
    """添加Episode到知识图谱"""
    pass
```

#### 4. 文档字符串
```python
def search_episodes(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    搜索Episodes - 改进版本，提供更友好的用户体验
    
    Args:
        query: 搜索查询字符串
        num_results: 返回结果数量，默认5个
        
    Returns:
        格式化的搜索结果列表，包含id、name、content等字段
        
    Raises:
        Exception: 搜索过程中的任何异常
    """
    pass
```

### 错误处理规范

#### 1. 异常处理模式
```python
async def add_episode(self, name: str, episode_body: str) -> Dict[str, Any]:
    """标准异常处理模式"""
    try:
        # 主要逻辑
        result = await self.graphiti.add_episode(...)
        logger.info(f"✅ 成功添加Episode: {name}")
        return {"success": True, "data": result}
        
    except Exception as e:
        logger.error(f"❌ 添加Episode失败: {e}")
        return {
            "success": False,
            "message": f"添加Episode失败: {str(e)}",
            "error": str(e)
        }
```

#### 2. 日志记录规范
```python
import logging

logger = logging.getLogger("GraphitiMCP")

# 成功操作：INFO级别，使用✅
logger.info(f"✅ 成功添加Episode: {name}")

# 警告：WARNING级别，使用⚠️
logger.warning(f"⚠️ Graphiti未初始化，使用模拟模式")

# 错误：ERROR级别，使用❌
logger.error(f"❌ 添加Episode失败: {e}")

# 调试：DEBUG级别，使用🔍
logger.debug(f"🔍 搜索参数: query={query}, num_results={num_results}")
```

### API设计规范

#### 1. 响应格式标准化
```python
# 成功响应
{
    "success": True,
    "message": "操作成功",
    "data": {...},
    "timestamp": "2026-01-04T07:30:00Z"
}

# 错误响应
{
    "success": False,
    "message": "错误描述",
    "error": "详细错误信息",
    "error_code": "ERROR_CODE",
    "timestamp": "2026-01-04T07:30:00Z"
}
```

#### 2. MCP协议兼容性
```python
# 工具定义标准格式
{
    "name": "tool_name",
    "description": "工具描述（中文）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "参数描述"},
            "param2": {"type": "integer", "description": "参数描述", "default": 5}
        },
        "required": ["param1"]
    }
}
```

## 🧪 测试规范

### 1. 测试文件命名
```
test_功能模块.py          # 单元测试
test_integration_*.py    # 集成测试
test_e2e_*.py           # 端到端测试
debug_*.py              # 调试脚本
```

### 2. 测试函数结构
```python
def test_add_episode_success():
    """测试成功添加Episode"""
    # Arrange - 准备测试数据
    name = "测试Episode"
    episode_body = "测试内容"
    
    # Act - 执行操作
    result = await wrapper.add_episode(name, episode_body)
    
    # Assert - 验证结果
    assert result["success"] is True
    assert result["episode_id"] is not None
    assert "测试Episode" in result["name"]
```

### 3. 测试覆盖要求
- **单元测试**：核心函数覆盖率 > 80%
- **集成测试**：主要API端点全覆盖
- **端到端测试**：关键用户场景全覆盖

### 4. 测试数据管理
```python
# 测试数据应该是可预测和可重复的
TEST_USER_INFO = {
    "name": "测试用户",
    "episode_body": "测试用户是一名软件工程师",
    "episode_type": "text",
    "source_description": "单元测试"
}

# 使用fixtures管理测试环境
@pytest.fixture
async def graphiti_wrapper():
    wrapper = GraphitiWrapper()
    await wrapper.initialize()
    yield wrapper
    await wrapper.close()
```

## 🔧 开发环境规范

### 1. 开发工具链
```bash
# 必需工具
pip install black          # 代码格式化
pip install flake8         # 代码检查
pip install mypy           # 类型检查
pip install pytest         # 测试框架
pip install pytest-asyncio # 异步测试支持
```

### 2. 预提交检查
```bash
# 代码格式化
black mcp_server/

# 代码检查
flake8 mcp_server/ --max-line-length=88

# 类型检查
mypy mcp_server/

# 运行测试
pytest tests/ -v
```

### 3. Docker开发环境
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  mcp-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - ./mcp_server:/app/mcp_server
      - ./tests:/app/tests
    environment:
      - PYTHONPATH=/app
    command: python -m pytest tests/ -v --watch
```

## 📁 项目结构规范

### 1. 目录组织
```
project_code/graphiti-demo/
├── mcp_server/              # 主应用代码
│   ├── __init__.py
│   ├── graphiti_mcp_server.py
│   ├── models/              # 数据模型
│   ├── services/            # 业务逻辑
│   ├── utils/               # 工具函数
│   └── config.py            # 配置管理
├── tests/                   # 测试代码
│   ├── unit/                # 单元测试
│   ├── integration/         # 集成测试
│   └── e2e/                 # 端到端测试
├── docs/                    # 文档
├── scripts/                 # 脚本工具
├── config/                  # 配置文件
└── docker/                  # Docker相关文件
```

### 2. 配置管理
```python
# config.py
import os
from typing import Optional

class Config:
    """配置管理类"""
    
    # API配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.laozhang.ai/v1")
    
    # 数据库配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    
    # 应用配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    @classmethod
    def validate(cls) -> None:
        """验证配置完整性"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
```

## 🚀 部署规范

### 1. 环境分离
```bash
# 开发环境
.env.development

# 测试环境
.env.testing

# 生产环境
.env.production
```

### 2. Docker镜像规范
```dockerfile
# 多阶段构建
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY mcp_server/ ./mcp_server/
EXPOSE 8000
CMD ["python", "-m", "mcp_server.graphiti_mcp_server"]
```

### 3. 健康检查
```python
@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查数据库连接
        db_status = await check_database_connection()
        
        # 检查外部API
        api_status = await check_external_api()
        
        return {
            "status": "healthy" if db_status and api_status else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "database": db_status,
                "external_api": api_status
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

## 📊 监控和日志规范

### 1. 日志格式
```python
import structlog

logger = structlog.get_logger("GraphitiMCP")

# 结构化日志
logger.info(
    "episode_added",
    episode_id=episode_id,
    user_id=user_id,
    content_length=len(episode_body),
    processing_time_ms=processing_time
)
```

### 2. 性能监控
```python
import time
from functools import wraps

def monitor_performance(func):
    """性能监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            logger.info(f"⏱️ {func.__name__} 执行时间: {duration:.2f}ms")
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"❌ {func.__name__} 执行失败: {e}, 耗时: {duration:.2f}ms")
            raise
    return wrapper
```

## 🔒 安全规范

### 1. 输入验证
```python
from pydantic import BaseModel, validator

class EpisodeRequest(BaseModel):
    name: str
    episode_body: str
    episode_type: str = "text"
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Episode name cannot be empty')
        if len(v) > 200:
            raise ValueError('Episode name too long')
        return v.strip()
    
    @validator('episode_body')
    def validate_body(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Episode body cannot be empty')
        if len(v) > 10000:
            raise ValueError('Episode body too long')
        return v.strip()
```

### 2. 敏感信息处理
```python
import re

def sanitize_log_message(message: str) -> str:
    """清理日志中的敏感信息"""
    # 隐藏API密钥
    message = re.sub(r'api[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9]+)', 
                     r'api_key: "***"', message, flags=re.IGNORECASE)
    
    # 隐藏密码
    message = re.sub(r'password["\s]*[:=]["\s]*([^\s"]+)', 
                     r'password: "***"', message, flags=re.IGNORECASE)
    
    return message
```

## 📝 文档规范

### 1. API文档
- 使用FastAPI自动生成的OpenAPI文档
- 每个端点必须有详细的描述和示例
- 错误响应必须有明确的错误码和说明

### 2. 代码文档
- 所有公共函数必须有docstring
- 复杂逻辑必须有行内注释
- 重要的设计决策必须有文档说明

### 3. 变更日志
```markdown
# CHANGELOG.md

## [1.2.0] - 2026-01-04

### Added
- 新增搜索结果格式化功能
- 支持中文内容类型标识

### Changed
- 优化搜索结果排序逻辑
- 改进错误处理机制

### Fixed
- 修复JSON序列化问题
- 修复MCP协议兼容性问题

### Deprecated
- 旧版搜索API将在v2.0中移除
```

## 🔄 版本管理规范

### 1. 分支策略
```
main          # 生产环境代码
develop       # 开发环境代码
feature/*     # 功能分支
hotfix/*      # 紧急修复分支
release/*     # 发布分支
```

### 2. 提交信息规范
```
feat: 添加新的搜索结果格式化功能
fix: 修复JSON序列化问题
docs: 更新API文档
test: 添加搜索功能单元测试
refactor: 重构错误处理逻辑
style: 代码格式化
chore: 更新依赖版本
```

### 3. 版本号规范
- 遵循语义化版本控制 (SemVer)
- 格式：MAJOR.MINOR.PATCH
- 示例：1.2.3

---

**规范执行**: 所有团队成员必须严格遵循本规范，代码审查时将检查规范遵循情况。