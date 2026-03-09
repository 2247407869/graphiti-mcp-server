const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const MCP_SERVER_URL = process.env.MCP_SERVER_URL || 'http://localhost:8000';

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// 健康检查
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// 代理MCP服务器请求
app.post('/api/mcp/*', async (req, res) => {
    try {
        const mcpPath = req.path.replace('/api/mcp', '');
        const response = await axios.post(`${MCP_SERVER_URL}${mcpPath}`, req.body, {
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: 10000
        });
        res.json(response.data);
    } catch (error) {
        console.error('MCP请求失败:', error.message);
        res.status(500).json({
            error: 'MCP服务器请求失败',
            message: error.message,
            suggestion: '请检查MCP服务器是否正常运行'
        });
    }
});

// 获取MCP服务器状态
app.get('/api/status', async (req, res) => {
    try {
        const response = await axios.get(`${MCP_SERVER_URL}/health`, { timeout: 5000 });
        res.json({
            mcp_server: 'connected',
            mcp_url: MCP_SERVER_URL,
            mcp_status: response.data
        });
    } catch (error) {
        res.json({
            mcp_server: 'disconnected',
            mcp_url: MCP_SERVER_URL,
            error: error.message
        });
    }
});

// 主页
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Graphiti演示Web界面运行在端口 ${PORT}`);
    console.log(`MCP服务器地址: ${MCP_SERVER_URL}`);
});