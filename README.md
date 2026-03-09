# Graphiti MCP Server

A powerful Model Context Protocol (MCP) server that supercharges AI agents with **Long-Term Memory (LTM)** capabilities, built on top of the [Graphiti](https://github.com/getZep/graphiti) Knowledge Graph engine and FalkorDB.

This project enables AI assistants (like LobeChat or Claude Desktop) to dynamically build, query, and reason over a persistent semantic knowledge graph of their interactions with users.

## 🌟 Key Features

- **🧠 True Long-Term Memory**: Dynamically extracts entities and relationships from conversations to build a persistent knowledge graph.
- **🔍 Semantic Search & Retrieval**: Uses vector embeddings to retrieve highly relevant facts, edges, and episodes based on contextual similarity.
- **🔌 Full MCP Compliance**: Exposes standard MCP tools (`add_episode`, `search`, `get_episodes`) for seamless plug-and-play integration with MCP clients like LobeChat.
- **🌐 OpenAI API Native**: Uses standard OpenAI-compatible endpoints for LLMs and Embeddings (configured for direct integration with services like Laozhang API or DeepSeek).
- **🐳 Docker Native**: Ships with a fully containerized setup (`docker-compose`) including the MCP Server, FalkorDB, and Web UI.
- **🇨🇳 Optimized for Chinese**: Enhances semantic extraction and edge formatting specifically for Chinese natural language processing, making graph queries human-readable.

## 🚀 Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- An OpenAI-compatible API Key (e.g., OpenAI, DeepSeek, or Laozhang API)

### 2. Configuration

Clone the repository and configure your environment variables:

```bash
# Copy the example env file
cp .env.example .env
```

Edit the `.env` file to include your API credentials:
```env
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Start the Ecosystem

Deploy the MCP server and FalkorDB using Docker Compose:

```bash
# Start all services in the background
docker-compose up -d

# Check the status of the containers
docker-compose ps
```

### 4. Service Endpoints

Once running, the following services are available:

- **MCP Server Endpoint**: `http://localhost:8000`
- **FalkorDB (Knowledge Graph)**: `localhost:6379`
- **LobeChat Interface**: `http://localhost:3210`
- **Demo Web Dashboard**: `http://localhost:3000`

## 🛠️ MCP Tools

The server exposes the following MCP tools to the AI Client:

| Tool Name | Description |
|-----------|-------------|
| `add_episode` | Ingests a new conversation or fact. The agent passes text, and Graphiti extracts nodes/edges automatically. |
| `search` | Semantically searches the knowledge graph for relevant facts based on a query string. |
| `get_episodes` | Retrieves chronologically stored conversation episodes for raw context recall. |

### Example Interaction Flow

1. **User**: "Please remember that I am currently researching Edge Computing and Serverless Architecture."
2. **AI Assistant**: *Calls `add_episode` tool with the text.*
3. *[Graphiti silently builds nodes for "User", "Edge Computing", and creates an "is researching" edge.]*
4. **User** *(days later)*: "What was I researching?"
5. **AI Assistant**: *Calls `search` tool with the query "User research interests". Returns the graph edges. Responds to user contextually.*

## 📋 Health Checks & API Debugging

You can manually verify that the MCP server is running correctly:

```bash
# Check Server Health
curl http://localhost:8000/health

# List available MCP Tools
curl http://localhost:8000/tools/list
```

## 📄 License

This project is licensed under the MIT License.
