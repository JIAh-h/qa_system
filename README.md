# 基于Transformer的问答系统

这是一个使用Transformer模型和知识库实现的问答系统，能够根据上下文回答用户问题。

## 系统特点

1. **本地模型推理**：使用Transformer架构的预训练模型进行本地推理，不依赖外部API
2. **基于知识库的问答**：从自定义知识库中检索相关信息回答问题，而不是依赖预训练模型的参数知识
3. **可训练**：支持使用自定义数据或自动生成的合成数据训练模型
4. **可扩展知识库**：可以随时添加新文档到知识库中

## 系统架构

- **前端**：基于HTML、CSS和JavaScript的简单Web界面
- **后端**：使用Django框架
- **核心组件**：
  - `TransformerQA`：基于Transformer的问答模型
  - `KnowledgeManager`：知识库管理和检索系统
  - 训练和推理管道

## 安装和运行

### 环境要求

- Python 3.8+
- Django 4.0+
- PyTorch 1.10+
- Transformers 4.15+
- 其他依赖请见requirements.txt

### 安装步骤

1. 克隆仓库
```
git clone <repository-url>
cd qa_system
```

2. 创建并激活虚拟环境（推荐）
```
python -m venv .venv
# 在Windows上
.venv\Scripts\activate
# 在Linux/Mac上
source .venv/bin/activate
```

3. 安装依赖
```
pip install -r requirements.txt
```

4. 准备知识库
```
# 在项目根目录下创建knowledge_base文件夹
mkdir -p ../knowledge_base
# 将您的文档(.txt, .md)放入此文件夹
```

5. 运行系统
```
python manage.py migrate
python manage.py runserver
```

6. 访问系统
在浏览器中访问 http://127.0.0.1:8000/

## 使用指南

### 提问

在输入框中输入您的问题，点击发送按钮或按回车键提交。系统会：
1. 在知识库中找到与您问题相关的内容
2. 根据上下文生成答案
3. 提供展示参考上下文的选项

### 训练模型

点击"训练模型"按钮可以使用知识库中的文档自动生成训练数据并训练模型。训练过程可能需要一些时间，取决于您的硬件配置。

### 更新知识库

添加新文档到knowledge_base文件夹后，点击"更新知识库"按钮刷新系统的知识。

## 自定义和扩展

### 添加自定义知识

将TXT或MD格式的文档添加到knowledge_base文件夹中，然后点击"更新知识库"按钮。

### 调整模型参数

打开`myapp/views.py`文件，修改MODEL_CONFIG字典中的配置：
```python
MODEL_CONFIG = {
    'MODEL_NAME': 'distilbert-base-uncased',  # 可以改为其他Transformer模型
    'MODEL_PATH': os.path.join(settings.BASE_DIR, 'qa_model'),
    'KB_DIR': os.path.join(settings.BASE_DIR, '..', 'knowledge_base'),
}
```

## 许可

本项目使用MIT许可证。 