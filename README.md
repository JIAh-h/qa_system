# 智能问答系统

这是一个基于BERT的智能问答系统，支持知识库管理、模型训练和问答功能。

## 功能特点

- 基于BERT的问答模型
- 知识库管理
- 模型训练和更新
- 文档缓存机制
- 安全访问控制
- 配置管理
- 日志系统
- 模型预训练支持
- 多模型集成
- 实时问答响应
- 知识库增量更新
- 模型性能评估
- 分布式训练支持

## 系统要求

- Python 3.8+
- Django 3.2+
- PyTorch 1.10+
- Transformers 4.15+
- NumPy 1.19+
- jieba
- sentence-transformers 2.2+
- tqdm
- django-cors-headers
- scikit-learn
- pandas
- matplotlib
- tensorboard

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd qa_system
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
export DJANGO_SECRET_KEY='your-secret-key'
export ENCRYPTION_KEY='your-encryption-key'
export CUDA_VISIBLE_DEVICES='0'  # 设置GPU设备
```

## 配置说明

系统配置位于 `myapp/config.py`，包含以下主要配置项：

### 模型配置
- MODEL_NAME: 使用的预训练模型名称
- MODEL_PATH: 模型保存路径
- KB_DIR: 知识库目录
- MAX_LENGTH: 最大序列长度
- TOP_K: 返回的最相关文档数
- THRESHOLD: 相似度阈值
- BATCH_SIZE: 训练批次大小
- LEARNING_RATE: 学习率
- EPOCHS: 训练轮数
- WARMUP_STEPS: 预热步数
- GRADIENT_ACCUMULATION_STEPS: 梯度累积步数
- MAX_GRAD_NORM: 梯度裁剪阈值
- WEIGHT_DECAY: 权重衰减率

### 系统配置
- LOG_DIR: 日志目录
- CACHE_DIR: 缓存目录
- MAX_DOCUMENTS: 最大文档数
- CHUNK_SIZE: 文档分块大小
- CHUNK_OVERLAP: 分块重叠大小
- MAX_MEMORY_USAGE: 最大内存使用比例
- REQUEST_TIMEOUT: 请求超时时间
- RATE_LIMIT: 每分钟最大请求数
- GPU_MEMORY_FRACTION: GPU内存使用比例
- NUM_WORKERS: 数据加载线程数
- PIN_MEMORY: 是否使用固定内存

### 安全配置
- ALLOWED_HOSTS: 允许访问的主机列表
- SECRET_KEY: Django密钥
- DEBUG: 调试模式
- ENCRYPTION_KEY: 数据加密密钥
- TOKEN_EXPIRE_TIME: Token过期时间
- MAX_LOGIN_ATTEMPTS: 最大登录尝试次数

## 使用说明

### 1. 初始化系统

```python
from myapp.views import init_qa_system

# 初始化系统
init_qa_system()
```

### 2. 问答功能

```python
from myapp.knowledge_manager import knowledge_manager

# 生成答案
answer = knowledge_manager.generate_answer("你的问题")
```

### 3. 模型训练

```python
from myapp.views import train_new_model

# 训练新模型
train_new_model(
    train_data_path="path/to/train_data",
    val_data_path="path/to/val_data",
    model_config={
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3
    }
)
```

### 4. 知识库更新

```python
from myapp.views import update_knowledge_view

# 更新知识库
update_knowledge_view(
    new_docs_path="path/to/new_docs",
    incremental=True
)
```

### 5. 模型评估

```python
from myapp.views import evaluate_model

# 评估模型性能
metrics = evaluate_model(
    test_data_path="path/to/test_data",
    metrics=["accuracy", "f1", "precision", "recall"]
)
```

## API接口

### 1. 问答接口
- 端点：`/qa/`
- 方法：POST
- 请求体：
```json
{
    "question": "你的问题",
    "top_k": 3,
    "threshold": 0.7
}
```
- 响应：
```json
{
    "answer": "系统回答",
    "confidence": 0.95,
    "sources": ["文档1", "文档2"],
    "status": "success"
}
```

### 2. 模型训练接口
- 端点：`/train/`
- 方法：POST
- 功能：支持上传训练数据集或使用默认数据集
- 请求体：
```json
{
    "train_data": "训练数据路径",
    "val_data": "验证数据路径",
    "model_config": {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3
    }
}
```

### 3. 知识库更新接口
- 端点：`/update_knowledge/`
- 方法：POST
- 功能：重新加载知识库
- 请求体：
```json
{
    "new_docs_path": "新文档路径",
    "incremental": true
}
```

### 4. 模型评估接口
- 端点：`/evaluate/`
- 方法：POST
- 功能：评估模型性能
- 请求体：
```json
{
    "test_data_path": "测试数据路径",
    "metrics": ["accuracy", "f1", "precision", "recall"]
}
```

## 安全特性

1. 访问控制
   - 主机白名单
   - 请求频率限制
   - 数据加密
   - Token认证
   - 登录尝试限制

2. 缓存机制
   - 模型缓存
   - 文档缓存
   - 自动过期清理
   - 内存使用限制
   - 缓存预热

3. 错误处理
   - 异常捕获
   - 日志记录
   - 优雅降级
   - 错误重试
   - 超时处理

## 目录结构

```
qa_system/
├── myapp/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── cache_manager.py   # 缓存管理
│   ├── security.py        # 安全控制
│   ├── knowledge_manager.py # 知识管理
│   ├── train_model.py     # 模型训练
│   ├── evaluate.py        # 模型评估
│   ├── utils.py          # 工具函数
│   └── views.py          # 视图处理
├── knowledge_base/       # 知识库目录
├── models/              # 模型目录
├── logs/               # 日志目录
├── cache/             # 缓存目录
├── tests/             # 测试目录
├── docs/              # 文档目录
└── requirements.txt   # 依赖列表
```

## 注意事项

1. 首次运行前请确保：
   - 配置正确的环境变量
   - 创建必要的目录结构
   - 准备训练数据集
   - 检查GPU可用性
   - 设置适当的内存限制

2. 安全建议：
   - 定期更新密钥
   - 限制访问IP
   - 监控系统日志
   - 定期备份数据
   - 使用HTTPS

3. 性能优化：
   - 适当调整缓存大小
   - 控制文档数量
   - 定期清理缓存
   - 使用GPU加速
   - 优化数据加载

4. 模型训练建议：
   - 使用验证集监控训练
   - 实现早停机制
   - 保存最佳模型
   - 使用学习率调度
   - 实现梯度累积

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License 