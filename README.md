# Kimi Knowledge Base

基于 Kimi2 API 构建的本地知识库系统，提供安全、高效的文档处理和智能问答服务。

## 功能特性

- 📄 多格式文档处理（PDF、Word、PPT等）
- 🔍 混合检索（向量检索 + 关键词检索）
- 🤖 基于 Kimi2 API 的智能问答
- 🔒 严格基于本地文档的约束性回答
- 📊 向量化存储和语义搜索
- 🚀 高性能缓存系统
- 🐳 Docker 容器化部署

## 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose
- Tesseract OCR

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd kimi-knowledge-base
```

2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，设置必要的配置
```

3. 使用 Docker 启动服务
```bash
make docker-up
```

4. 或者本地开发
```bash
make dev-install
make run
```

### API 访问

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 系统状态: http://localhost:8000/api/v1/system/status
- 性能监控: http://localhost:8000/api/v1/system/performance
- Qdrant 管理界面: http://localhost:6333/dashboard

### 主要API端点

#### 文档管理
- `POST /api/v1/documents/upload` - 上传文档
- `GET /api/v1/documents` - 列出文档
- `GET /api/v1/documents/{doc_id}` - 获取文档信息
- `DELETE /api/v1/documents/{doc_id}` - 删除文档

#### 搜索功能
- `POST /api/v1/search` - 混合搜索
- `POST /api/v1/search/vector` - 向量搜索
- `POST /api/v1/search/bm25` - 关键词搜索

#### 问答功能
- `POST /api/v1/qa` - 智能问答

#### 系统监控
- `GET /api/v1/system/status` - 系统状态
- `GET /api/v1/system/performance` - 性能指标
- `GET /api/v1/system/cache/stats` - 缓存统计

## 开发指南

### 项目结构

```
src/
├── api/           # FastAPI 应用
├── config/        # 配置管理
├── models/        # 数据模型
├── services/      # 业务服务
└── utils/         # 工具函数

tests/             # 测试文件
data/              # 数据存储
logs/              # 日志文件
```

### 开发命令

```bash
make help          # 查看所有可用命令
make test          # 运行测试
make lint          # 代码检查
make format        # 代码格式化
make clean         # 清理临时文件
```

## 配置说明

主要配置项在 `.env` 文件中：

- `MOONSHOT_API_KEY`: Kimi2 API 密钥
- `QDRANT_HOST`: 向量数据库地址
- `UPLOAD_DIR`: 文档上传目录
- `LOG_LEVEL`: 日志级别

## 部署

### Docker 部署

```bash
# 构建镜像
make docker-build

# 启动服务
make docker-up

# 查看日志
make docker-logs

# 停止服务
make docker-down
```

### 监控（可选）

启用 Prometheus 和 Grafana 监控：

```bash
make monitoring-up
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## 许可证

MIT License