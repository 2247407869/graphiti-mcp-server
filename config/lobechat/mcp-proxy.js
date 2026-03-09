#!/usr/bin/env node
/**
 * Graphiti MCP代理服务器
 * 将标准MCP协议请求转发到我们的HTTP MCP服务器
 */

const http = require('http');
const { URL } = require('url');

const MCP_SERVER_URL = process.env.MCP_SERVER_URL || 'http://graphiti-mcp:8000';

class GraphitiMCPProxy {
    constructor() {
        this.setupStdio();
        this.sendInitialCapabilities();
    }

    setupStdio() {
        let buffer = '';
        
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', (chunk) => {
            buffer += chunk;
            
            // 处理完整的JSON-RPC消息
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // 保留不完整的行
            
            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const message = JSON.parse(line);
                        this.handleMessage(message);
                    } catch (error) {
                        this.sendError(-32700, 'Parse error', null);
                    }
                }
            }
        });

        process.stdin.on('end', () => {
            process.exit(0);
        });

        // 处理进程信号
        process.on('SIGINT', () => process.exit(0));
        process.on('SIGTERM', () => process.exit(0));
    }

    sendInitialCapabilities() {
        // 发送初始化能力信息
        this.sendNotification('initialized', {
            protocolVersion: "2024-11-05",
            capabilities: {
                tools: {}
            },
            serverInfo: {
                name: "graphiti-memory",
                version: "1.0.0"
            }
        });
    }

    async handleMessage(message) {
        try {
            const { id, method, params } = message;

            switch (method) {
                case 'initialize':
                    this.sendResponse(id, {
                        protocolVersion: "2024-11-05",
                        capabilities: {
                            tools: {}
                        },
                        serverInfo: {
                            name: "graphiti-memory",
                            version: "1.0.0"
                        }
                    });
                    break;

                case 'tools/list':
                    const tools = await this.getTools();
                    this.sendResponse(id, { tools });
                    break;

                case 'tools/call':
                    const result = await this.callTool(params);
                    this.sendResponse(id, result);
                    break;

                case 'notifications/initialized':
                    // 忽略初始化通知
                    break;

                default:
                    this.sendError(-32601, 'Method not found', id);
            }
        } catch (error) {
            this.sendError(-32603, 'Internal error: ' + error.message, message.id);
        }
    }

    async getTools() {
        try {
            const response = await this.httpRequest('/tools/list', 'GET');
            return response.tools || [];
        } catch (error) {
            return [
                {
                    name: "add_episode",
                    description: "向知识图谱添加一个Episode（记忆片段）",
                    inputSchema: {
                        type: "object",
                        properties: {
                            name: { type: "string", description: "Episode名称" },
                            episode_body: { type: "string", description: "Episode内容" },
                            episode_type: { type: "string", description: "Episode类型", default: "text" },
                            source_description: { type: "string", description: "来源描述", default: "MCP" }
                        },
                        required: ["name", "episode_body"]
                    }
                },
                {
                    name: "search",
                    description: "搜索知识图谱中的Episodes",
                    inputSchema: {
                        type: "object",
                        properties: {
                            query: { type: "string", description: "搜索查询" },
                            num_results: { type: "integer", description: "返回结果数量", default: 5 }
                        },
                        required: ["query"]
                    }
                },
                {
                    name: "get_episodes",
                    description: "获取Episodes列表",
                    inputSchema: {
                        type: "object",
                        properties: {
                            limit: { type: "integer", description: "返回数量限制", default: 100 }
                        }
                    }
                }
            ];
        }
    }

    async callTool(params) {
        try {
            const response = await this.httpRequest('/tools/call', 'POST', params);
            return response.result || response;
        } catch (error) {
            return { 
                error: {
                    code: -1,
                    message: error.message
                }
            };
        }
    }

    httpRequest(path, method, data = null) {
        return new Promise((resolve, reject) => {
            const url = new URL(path, MCP_SERVER_URL);
            const options = {
                hostname: url.hostname,
                port: url.port,
                path: url.pathname,
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                },
                timeout: 10000
            };

            const req = http.request(options, (res) => {
                let body = '';
                res.on('data', (chunk) => body += chunk);
                res.on('end', () => {
                    try {
                        if (res.statusCode >= 200 && res.statusCode < 300) {
                            const response = JSON.parse(body);
                            resolve(response);
                        } else {
                            reject(new Error(`HTTP ${res.statusCode}: ${body}`));
                        }
                    } catch (error) {
                        reject(new Error('Invalid JSON response: ' + body));
                    }
                });
            });

            req.on('error', (error) => {
                reject(new Error('Network error: ' + error.message));
            });

            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });

            if (data) {
                req.write(JSON.stringify(data));
            }
            req.end();
        });
    }

    sendResponse(id, result) {
        const response = {
            jsonrpc: "2.0",
            id: id,
            result: result
        };
        process.stdout.write(JSON.stringify(response) + '\n');
    }

    sendError(code, message, id) {
        const response = {
            jsonrpc: "2.0",
            id: id,
            error: {
                code: code,
                message: message
            }
        };
        process.stdout.write(JSON.stringify(response) + '\n');
    }

    sendNotification(method, params) {
        const notification = {
            jsonrpc: "2.0",
            method: method,
            params: params
        };
        process.stdout.write(JSON.stringify(notification) + '\n');
    }
}

// 启动代理
if (require.main === module) {
    new GraphitiMCPProxy();
}