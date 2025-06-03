import os
import glob
import json
import logging
import threading
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer, BertForQuestionAnswering, pipeline
import torch
import time
import traceback
import jieba
from .config import config
from .cache_manager import model_cache, document_cache
from .security import access_control

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeManager:
    def __init__(self):
        """初始化知识库管理器"""
        self.qa_pairs = []  # 存储问答对
        self.documents = {}  # 存储文档
        self.embeddings = {}  # 存储文档嵌入
        self.lock = threading.Lock()
        self.model_config = config.get_model_config()
        self.system_config = config.get_system_config()
        self._model = None
        self._tokenizer = None
        self._qa_pipeline = None
        self._question_embeddings = {}  # 缓存问题的嵌入向量
        
    def _get_model_and_tokenizer(self):
        """获取模型和分词器（带缓存）"""
        if self._model is None or self._tokenizer is None:
            logger.info("首次加载模型和分词器...")
            self._model, self._tokenizer = model_cache.get_model()
            logger.info("模型和分词器加载完成")
        return self._model, self._tokenizer
        
    def _get_qa_pipeline(self):
        """获取QA pipeline（带缓存）"""
        if self._qa_pipeline is None:
            logger.info("首次创建QA pipeline...")
            model, tokenizer = self._get_model_and_tokenizer()
            self._qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("QA pipeline创建完成")
        return self._qa_pipeline
        
    def _get_question_embedding(self, question: str) -> np.ndarray:
        """获取问题的语义表示（带缓存）"""
        try:
            # 检查缓存
            if question in self._question_embeddings:
                logger.info("使用缓存的问题嵌入")
                return self._question_embeddings[question]
                
            logger.info("开始计算问题嵌入...")
            model, tokenizer = self._get_model_and_tokenizer()
            
            inputs = tokenizer(
                question,
                max_length=self.model_config['MAX_LENGTH'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model.bert(**inputs)
                embedding = outputs.last_hidden_state[0, 0, :].numpy()
                embedding = embedding / np.linalg.norm(embedding)
                
            # 存入缓存
            self._question_embeddings[question] = embedding
            logger.info("问题嵌入计算完成并缓存")
            return embedding
                
        except Exception as e:
            logger.error(f"获取问题嵌入时出错: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return np.zeros(768)
            
    def generate_answer(self, question: str) -> str:
        """生成答案"""
        try:
            if not question or not question.strip():
                return "问题不能为空"
                
            logger.info(f"开始处理问题: {question}")
            
            # 首先在数据集中查找相似问题
            best_similarity = 0.3
            best_answer = None
            best_question = None
            
            # 获取问题的语义表示（使用缓存）
            question_embedding = self._get_question_embedding(question)
            
            # 计算相似度
            for q, a in self.qa_pairs:
                similarity = self._calculate_semantic_similarity(question, q, question_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_answer = a
                    best_question = q
                    
            # 如果在数据集中找到相似问题，返回对应的答案
            if best_answer:
                logger.info(f"找到相似问题: {best_question} (相似度: {best_similarity:.2f})")
                return best_answer
                
            # 如果没有找到相似问题，使用模型生成答案
            qa_pipeline = self._get_qa_pipeline()
            
            context = self._get_context_for_question(question)
            if not context:
                context = "这是一个智能问答系统。我可以回答各种问题。"
                
            result = qa_pipeline(
                question=question,
                context=context,
                max_answer_len=100,
                handle_impossible_answer=True
            )
            
            if result and 'answer' in result and result['answer']:
                return result['answer']
            else:
                return "抱歉，我暂时无法回答这个问题。请换个方式提问，或者询问其他问题。"
                
        except Exception as e:
            logger.error(f"生成答案时出错: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return "抱歉，生成答案时出现错误。"
            
    def load_documents_from_directory(self, directory_path: str) -> bool:
        """从目录加载文档"""
        try:
            logger.info(f"从目录加载文档: {directory_path}")
            
            if not os.path.exists(directory_path):
                logger.error(f"目录不存在: {directory_path}")
                return False
                
            # 获取所有文件
            files = []
            for ext in ['.txt', '.md', '.json']:
                files.extend([f for f in os.listdir(directory_path) if f.endswith(ext)])
                
            if not files:
                logger.warning(f"在目录 {directory_path} 中没有找到符合条件的文件")
                return False
                
            # 加载文档
            for file_name in files:
                file_path = os.path.join(directory_path, file_name)
                try:
                    if file_name.endswith('.json'):
                        # 处理JSON文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 处理对话列表
                            for conv in data:
                                if isinstance(conv, list) and len(conv) >= 2:
                                    # 将对话转换为问答对
                                    for i in range(0, len(conv)-1, 2):
                                        if i+1 < len(conv):
                                            self.qa_pairs.append((conv[i], conv[i+1]))
                                            # 加密敏感数据
                                            encrypted_q = access_control.encrypt_sensitive_data(conv[i])
                                            encrypted_a = access_control.encrypt_sensitive_data(conv[i+1])
                                            document_cache.set_document(f"qa_{len(self.qa_pairs)}", 
                                                                      json.dumps({"q": encrypted_q, "a": encrypted_a}))
                    else:
                        # 处理其他类型的文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                self.documents[file_name] = content
                                document_cache.set_document(file_name, content)
                                
                except Exception as e:
                    logger.error(f"加载文档失败 {file_path}: {str(e)}")
                    continue
                    
            logger.info(f"成功加载了 {len(self.documents)} 个文档")
            logger.info(f"从文档中提取了 {len(self.qa_pairs)} 个问答对")
            return True
            
        except Exception as e:
            logger.error(f"加载文档时出错: {str(e)}")
            return False
            
    def _calculate_semantic_similarity(self, q1: str, q2: str, q1_embedding: np.ndarray) -> float:
        """计算两个问题的语义相似度"""
        try:
            # 获取第二个问题的嵌入
            q2_embedding = self._get_question_embedding(q2)
            
            # 计算余弦相似度
            semantic_similarity = np.dot(q1_embedding, q2_embedding)
            
            # 计算词重叠相似度
            words1 = set(jieba.cut(q1))
            words2 = set(jieba.cut(q2))
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
            
            # 综合两种相似度
            final_similarity = 0.7 * semantic_similarity + 0.3 * word_overlap
            
            return final_similarity
            
        except Exception as e:
            logger.error(f"计算语义相似度时出错: {str(e)}")
        return 0.0

    def _get_context_for_question(self, question: str) -> str:
        """为问题获取相关上下文"""
        try:
            # 获取文档嵌入
            model, tokenizer = model_cache.get_model()
            
            # 编码问题
            question_embedding = self._get_question_embedding(question)
            
            # 计算相似度并获取最相关的文档
            similarities = []
            for doc_id, doc_content in self.documents.items():
                # 获取或计算文档嵌入
                doc_embedding = document_cache.get_embedding(doc_id)
                if doc_embedding is None:
                    doc_embedding = self._encode_text(doc_content, model, tokenizer)
                    document_cache.set_embedding(doc_id, doc_embedding)
                    
                # 计算相似度
                similarity = np.dot(question_embedding, doc_embedding)
                similarities.append((doc_id, similarity))
                
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 获取最相关的文档内容
            contexts = []
            for doc_id, _ in similarities[:self.model_config['TOP_K']]:
                contexts.append(self.documents[doc_id])
                
            return "\n".join(contexts)
            
        except Exception as e:
            logger.error(f"获取上下文时出错: {str(e)}")
            return ""
            
    def _encode_text(self, text: str, model, tokenizer) -> np.ndarray:
        """编码文本"""
        try:
            # 对文本进行编码
            inputs = tokenizer(
                text,
                max_length=self.model_config['MAX_LENGTH'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # 计算BERT嵌入
            with torch.no_grad():
                # 使用模型的encoder部分获取隐藏状态
                outputs = model.bert(**inputs)
                # 使用[CLS]标记的输出作为文本表示
                embedding = outputs.last_hidden_state[0, 0, :].numpy()
                # 归一化
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            
        except Exception as e:
            logger.error(f"编码文本时出错: {str(e)}")
            return np.zeros(768)  # 返回零向量作为后备方案

# 创建全局知识管理器实例
knowledge_manager = KnowledgeManager() 