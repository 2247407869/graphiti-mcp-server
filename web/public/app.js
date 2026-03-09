// Graphiti演示Web应用

class GraphitiDemo {
    constructor() {
        this.mcpServerUrl = '/api/mcp';
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkConnection();
        this.refreshStats();
    }

    bindEvents() {
        // 添加记忆表单
        document.getElementById('addEpisodeForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addEpisode();
        });

        // 搜索表单
        document.getElementById('searchForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.searchMemory();
        });

        // 刷新统计
        document.getElementById('refreshStats').addEventListener('click', () => {
            this.refreshStats();
        });

        // 回车键搜索
        document.getElementById('searchQuery').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.searchMemory();
            }
        });
    }

    async checkConnection() {
        const statusEl = document.getElementById('connectionStatus');
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.mcp_server === 'connected') {
                const mode = data.mcp_status?.mode || 'unknown';
                const modeText = mode === 'simulation' ? '(模拟模式)' : mode === 'real' ? '(真实模式)' : '';
                statusEl.textContent = `✅ MCP服务器已连接 ${modeText}`;
                statusEl.className = 'status connected';
            } else {
                statusEl.textContent = '❌ MCP服务器连接失败';
                statusEl.className = 'status disconnected';
            }
        } catch (error) {
            statusEl.textContent = '❌ 无法连接到服务器';
            statusEl.className = 'status disconnected';
        }
    }

    async addEpisode() {
        const name = document.getElementById('episodeName').value.trim();
        const content = document.getElementById('episodeContent').value.trim();
        const resultEl = document.getElementById('addResult');

        if (!content) {
            this.showError(resultEl, '请输入记忆内容');
            return;
        }

        try {
            // 直接调用MCP工具API
            const toolRequest = {
                name: 'add_episode',
                arguments: {
                    name: name || '未命名记忆',
                    episode_body: content,
                    episode_type: 'text',
                    source_description: 'Web界面添加'
                }
            };

            const response = await fetch(`${this.mcpServerUrl}/tools/call`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(toolRequest)
            });

            const data = await response.json();

            if (response.ok && data.result) {
                this.showSuccess(resultEl, '记忆添加成功！');
                document.getElementById('episodeName').value = '';
                document.getElementById('episodeContent').value = '';
                this.refreshStats();
            } else {
                this.showError(resultEl, data.error || '添加失败');
            }
        } catch (error) {
            this.showError(resultEl, `网络错误: ${error.message}`);
        }
    }

    async searchMemory() {
        const query = document.getElementById('searchQuery').value.trim();
        const resultsEl = document.getElementById('searchResults');
        const loadingEl = document.getElementById('searchLoading');

        if (!query) {
            this.showError(resultsEl, '请输入搜索内容');
            return;
        }

        loadingEl.style.display = 'block';
        resultsEl.innerHTML = '';

        try {
            // 直接调用MCP搜索工具
            const toolRequest = {
                name: 'search',
                arguments: {
                    query: query,
                    num_results: 10
                }
            };

            const response = await fetch(`${this.mcpServerUrl}/tools/call`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(toolRequest)
            });

            const data = await response.json();
            loadingEl.style.display = 'none';

            if (response.ok && data.result) {
                this.displaySearchResults(data.result, resultsEl);
            } else {
                this.showError(resultsEl, data.error || '搜索失败');
            }
        } catch (error) {
            loadingEl.style.display = 'none';
            this.showError(resultsEl, `网络错误: ${error.message}`);
        }
    }

    displaySearchResults(results, container) {
        if (!results || results.length === 0) {
            container.innerHTML = '<div class="result-item">未找到相关记忆</div>';
            return;
        }

        container.innerHTML = results.map(result => `
            <div class="result-item">
                <div class="result-score">相关度: ${(result.score || 0).toFixed(2)}</div>
                <strong>${result.name || '未命名'}</strong>
                <p>${result.content || result.episode_body || '无内容'}</p>
                <small style="color: #666;">
                    ${result.created_at ? new Date(result.created_at).toLocaleString() : ''}
                </small>
            </div>
        `).join('');
    }

    async refreshStats() {
        try {
            // 获取Episode统计
            const toolRequest = {
                name: 'get_episodes',
                arguments: {
                    limit: 1000  // 获取总数
                }
            };

            const response = await fetch(`${this.mcpServerUrl}/tools/call`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(toolRequest)
            });

            const data = await response.json();

            if (response.ok && data.result) {
                const count = Array.isArray(data.result) ? data.result.length : 0;
                document.getElementById('episodeCount').textContent = count;
                document.getElementById('systemStatus').textContent = '正常';
                document.getElementById('systemStatus').style.color = '#28a745';
            } else {
                document.getElementById('episodeCount').textContent = '?';
                document.getElementById('systemStatus').textContent = '异常';
                document.getElementById('systemStatus').style.color = '#dc3545';
            }
        } catch (error) {
            document.getElementById('episodeCount').textContent = '?';
            document.getElementById('systemStatus').textContent = '离线';
            document.getElementById('systemStatus').style.color = '#dc3545';
            console.error('统计刷新失败:', error);
        }
    }

    showSuccess(element, message) {
        element.innerHTML = `<div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 6px; margin-top: 10px;">${message}</div>`;
        setTimeout(() => element.innerHTML = '', 3000);
    }

    showError(element, message) {
        element.innerHTML = `<div class="error">${message}</div>`;
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new GraphitiDemo();
});