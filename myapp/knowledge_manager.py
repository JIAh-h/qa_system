import os
import glob
import logging
from typing import Dict, List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize
import time
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeManager:
    def __init__(self, encoder_model="sentence-transformers/all-MiniLM-L6-v2"):
        """初始化知识库管理器"""
        self.encoder_model = encoder_model
        self.documents = {}  # 文档ID到文档内容的映射
        self.embeddings = {}  # 文档ID到嵌入向量的映射
        self.document_chunks = {}  # 文档ID到文档分块的映射
        self.chunk_embeddings = {}  # 文档分块ID到嵌入向量的映射
        
        # 初始化编码器
        try:
            logger.info(f"加载编码器模型: {encoder_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.model = AutoModel.from_pretrained(encoder_model)
            
            # 设置设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用设备: {self.device}")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"初始化编码器失败: {str(e)}")
            # 使用后备模型
            try:
                logger.info("尝试使用后备模型: distilbert-base-uncased")
                self.encoder_model = "distilbert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
                self.model = AutoModel.from_pretrained(self.encoder_model)
                self.model.to(self.device)
            except Exception as e2:
                logger.error(f"加载后备模型也失败: {str(e2)}")
                raise
    
    def load_documents_from_directory(self, directory_path: str, extensions: List[str] = ['.txt', '.md']) -> Dict[str, str]:
        """从目录加载文档"""
        logger.info(f"从目录加载文档: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"目录不存在: {directory_path}")
            return {}
        
        # 构建文件匹配模式
        patterns = [os.path.join(directory_path, f"*{ext}") for ext in extensions]
        
        # 获取所有匹配的文件
        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(pattern))
        
        if not all_files:
            logger.warning(f"在目录 {directory_path} 中没有找到符合条件的文件")
            return {}
        
        # 加载文档
        loaded_docs = {}
        for file_path in all_files:
            doc_id = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if not content.strip():
                    logger.warning(f"文档 {doc_id} 为空")
                    continue
                loaded_docs[doc_id] = content
                logger.info(f"加载文档: {doc_id}, 长度: {len(content)} 字符")
            except Exception as e:
                logger.error(f"加载文档失败 {file_path}: {str(e)}")
        
        if not loaded_docs:
            logger.warning("没有成功加载任何文档")
        else:
            logger.info(f"成功加载了 {len(loaded_docs)} 个文档")
            
        self.documents.update(loaded_docs)
        return loaded_docs
    
    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 100) -> Dict[str, List[Tuple[str, str]]]:
        """将文档分块"""
        logger.info(f"将文档分块，块大小: {chunk_size}，重叠: {chunk_overlap}")
        
        if not self.documents:
            logger.warning("没有文档可供分块")
            return {}
            
        chunked_docs = {}
        
        for doc_id, content in self.documents.items():
            # 更智能的分块 - 尝试在段落或句子边界分割
            chunks = []
            
            # 首先按段落分割
            paragraphs = content.split('\n\n')
            current_chunk = ""
            current_start = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                # 如果当前段落加上已有内容超过了块大小，则创建新块
                if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
                    chunk_id = f"{doc_id}_chunk_{current_start}"
                    chunks.append((chunk_id, current_chunk))
                    
                    # 计算新的开始位置，考虑重叠
                    overlap_chars = min(chunk_overlap, len(current_chunk))
                    current_start = max(0, len(current_chunk) - overlap_chars)
                    current_chunk = current_chunk[-overlap_chars:] if overlap_chars > 0 else ""
                
                # 添加段落到当前块
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # 处理最后一个块
            if current_chunk:
                chunk_id = f"{doc_id}_chunk_{current_start}"
                chunks.append((chunk_id, current_chunk))
            
            chunked_docs[doc_id] = chunks
            logger.info(f"文档 {doc_id} 被分成了 {len(chunks)} 个块")
        
        # 更新文档分块
        self.document_chunks = chunked_docs
        
        # 展平所有块，方便后续处理
        all_chunks = {}
        for chunks in chunked_docs.values():
            for chunk_id, chunk_content in chunks:
                all_chunks[chunk_id] = chunk_content
        
        logger.info(f"总共生成了 {len(all_chunks)} 个文本块")
        return all_chunks
    
    def encode_text(self, text: str, max_retries: int = 3) -> np.ndarray:
        """使用预训练模型编码文本，获取嵌入向量"""
        # 切换到评估模式
        self.model.eval()
        
        # 对于过长的文本，截断处理
        if len(text) > 10000:
            logger.warning(f"文本过长 ({len(text)} 字符)，将被截断到前10000个字符")
            text = text[:10000]
        
        for retry in range(max_retries):
            try:
                # 对文本进行编码
                with torch.no_grad():
                    inputs = self.tokenizer(text, padding=True, truncation=True, 
                                           max_length=512, return_tensors="pt")
                    # 将输入移动到设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # 获取模型输出
                    outputs = self.model(**inputs)
                    
                    # 使用最后一层隐藏状态的平均值作为文本嵌入
                    attention_mask = inputs['attention_mask']
                    
                    # 对隐藏状态求平均
                    embeddings = outputs.last_hidden_state
                    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    masked_embeddings = embeddings * mask
                    summed = torch.sum(masked_embeddings, dim=1)
                    counts = torch.sum(attention_mask, dim=1, keepdim=True).float()
                    mean_pooled = summed / counts
                    
                    # 归一化嵌入向量
                    normalized_embeddings = normalize(mean_pooled, p=2, dim=1)
                    
                    # 转换为NumPy数组并返回
                    return normalized_embeddings.cpu().numpy()[0]
            except Exception as e:
                logger.error(f"编码失败 (尝试 {retry+1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    time.sleep(1)  # 等待一秒后重试
                else:
                    logger.error(f"编码失败，详细错误: {traceback.format_exc()}")
                    # 返回零向量作为后备方案
                    return np.zeros(768)  # 假设embedding维度为768
    
    def encode_documents(self) -> None:
        """为所有文档生成嵌入向量"""
        logger.info("为所有文档生成嵌入向量")
        
        if not self.documents:
            logger.warning("没有文档可供编码")
            return
        
        # 为完整文档生成嵌入
        for doc_id, content in self.documents.items():
            try:
                self.embeddings[doc_id] = self.encode_text(content)
                logger.info(f"编码文档: {doc_id}")
            except Exception as e:
                logger.error(f"编码文档 {doc_id} 失败: {str(e)}")
        
        # 为文档分块生成嵌入
        if not self.document_chunks:
            self.chunk_documents()
        
        for doc_id, chunks in self.document_chunks.items():
            for chunk_id, chunk_content in chunks:
                try:
                    self.chunk_embeddings[chunk_id] = self.encode_text(chunk_content)
                    logger.info(f"编码块: {chunk_id}")
                except Exception as e:
                    logger.error(f"编码块 {chunk_id} 失败: {str(e)}")
        
        logger.info(f"编码完成。文档数: {len(self.embeddings)}, 块数: {len(self.chunk_embeddings)}")
    
    def search(self, query: str, search_type: str = 'chunk', top_k: int = 3) -> List[Tuple[str, str, float]]:
        """搜索与查询最相关的文档或文档块"""
        logger.info(f"搜索查询: {query}")
        
        if not self.documents:
            logger.warning("没有文档可供搜索")
            return []
            
        # 如果还没有编码，进行编码
        if not self.embeddings and not self.chunk_embeddings:
            logger.info("尚未生成嵌入，现在进行编码...")
            self.encode_documents()
            
        if search_type == 'chunk' and not self.chunk_embeddings:
            logger.warning("没有块嵌入可用，将搜索整个文档")
            search_type = 'document'
            
        if search_type == 'document' and not self.embeddings:
            logger.warning("没有文档嵌入可用，将搜索文档块")
            search_type = 'chunk'
            
        # 编码查询
        query_embedding = self.encode_text(query)
        
        results = []
        
        # 根据搜索类型选择搜索范围
        if search_type == 'document':
            # 搜索整个文档
            for doc_id, embedding in self.embeddings.items():
                # 计算余弦相似度
                similarity = np.dot(query_embedding, embedding)
                results.append((doc_id, self.documents[doc_id], float(similarity)))
        else:
            # 搜索文档块
            for chunk_id, embedding in self.chunk_embeddings.items():
                # 计算余弦相似度
                similarity = np.dot(query_embedding, embedding)
                # 从chunk_id中提取原始文档ID
                doc_id = chunk_id.split('_chunk_')[0]
                # 获取块内容
                chunk_content = ""
                for doc_chunks in self.document_chunks.values():
                    for c_id, c_content in doc_chunks:
                        if c_id == chunk_id:
                            chunk_content = c_content
                            break
                    if chunk_content:
                        break
                
                results.append((doc_id, chunk_content, float(similarity)))
        
        # 按相似度排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        # 返回top_k个结果
        return results[:top_k]
    
    def get_context_for_question(self, question: str, top_k: int = 3) -> str:
        """为问题获取相关上下文"""
        if not question or not question.strip():
            logger.warning("提供的问题为空")
            return ""
            
        # 先尝试从datas.txt直接查找答案（如果是datas.txt文件）
        for doc_id, content in self.documents.items():
            if doc_id.lower() == "datas.txt":
                try:
                    # 解析问答对
                    parts = content.strip().split('问：')
                    parts = [p for p in parts if p.strip()]
                    
                    best_match = None
                    best_match_score = 0
                    best_answer = None
                    
                    # 预处理查询问题 - 去除停用词和标点符号，提取核心词
                    query_core = self._extract_core_terms(question)
                    
                    for part in parts:
                        qa_split = part.strip().split('答：', 1)
                        if len(qa_split) == 2:
                            q = qa_split[0].strip()
                            a = qa_split[1].strip()
                            
                            # 更强的问题相似度计算
                            score = self._calculate_question_similarity(q, question)
                            
                            # 如果找到更好的匹配
                            if score > best_match_score:
                                best_match = q
                                best_answer = a
                                best_match_score = score
                    
                    # 降低匹配阈值，但确保有一定相关性
                    if best_match and best_match_score > 0.3:
                        logger.info(f"从datas.txt中找到匹配问题: {best_match}, 相似度: {best_match_score}")
                        # 如果答案包含多个句子，只返回第一句
                        sentences = best_answer.split('。')
                        if len(sentences) > 1:
                            first_sentence = sentences[0].strip() + '。'
                            logger.info(f"只返回第一句答案: {first_sentence}")
                            return first_sentence
                        else:
                            # 确保答案干净（无问答标记）
                            return best_answer
                except Exception as e:
                    logger.error(f"解析datas.txt失败: {str(e)}")
        
        # 确保已编码文档
        if not self.chunk_embeddings and not self.embeddings:
            logger.info("文档尚未编码，正在进行编码...")
            if not self.documents:
                logger.warning("没有加载任何文档")
                return ""
            
            self.encode_documents()
        
        # 优先使用块搜索，如果没有块则使用文档搜索
        search_type = 'chunk' if self.chunk_embeddings else 'document'
        
        # 搜索与问题相关的文档块或文档
        relevant_items = self.search(question, search_type, top_k=1)  # 只获取最相关的一个结果
        
        if not relevant_items:
            logger.warning(f"没有找到与问题相关的{search_type}")
            return ""
        
        # 将相关块合并为上下文，但只使用最匹配的一个
        top_item = relevant_items[0]
        doc_id, content, similarity = top_item
        
        # 如果内容太长，尝试截取第一句或第一段
        if len(content) > 200:
            sentences = content.split('。')
            if len(sentences) > 1:
                content = sentences[0].strip() + '。'
            else:
                paragraphs = content.split('\n')
                if len(paragraphs) > 1:
                    content = paragraphs[0].strip()
                else:
                    # 如果没有明确的句子或段落分隔，取前200个字符
                    content = content[:200] + '...'
        
        logger.info(f"返回最匹配的上下文，相似度: {similarity}")
        return content
        
    def _extract_core_terms(self, text):
        """从文本中提取核心词汇，去除停用词和标点符号"""
        # 简单的中文停用词列表
        stop_words = ['的', '了', '和', '是', '在', '我', '你', '他', '她', '它', '这', '那', '哪', '什么', '怎么', '如何', '为什么']
        
        # 去除标点和特殊字符
        text = text.lower()
        for char in ',.?!;:，。？！；：""\'\'""《》【】()（）[]{}':
            text = text.replace(char, '')
            
        # 分词并去除停用词
        words = text.split()
        core_words = [w for w in words if w not in stop_words and len(w.strip()) > 0]
        
        return core_words
        
    def _calculate_question_similarity(self, q1, q2):
        """计算两个问题的相似度，综合多种匹配方式"""
        # 1. 准备文本
        q1 = q1.lower().strip()
        q2 = q2.lower().strip()
        
        # 2. 完全匹配检查
        if q1 == q2:
            return 1.0
            
        # 3. 包含关系检查 - 一个问题是另一个的子串
        if q1 in q2 or q2 in q1:
            # 计算更长问题中较短问题占的比例
            short_q, long_q = (q1, q2) if len(q1) <= len(q2) else (q2, q1)
            return 0.8 * (len(short_q) / len(long_q))
        
        # 4. 关键词匹配
        # 去除标点和停用词
        q1_core = self._extract_core_terms(q1)
        q2_core = self._extract_core_terms(q2)
        
        # 如果没有提取到核心词，回退到原始分词
        if not q1_core:
            q1_core = q1.split()
        if not q2_core:
            q2_core = q2.split()
        
        # 计算Jaccard相似度（集合交集/并集）
        if q1_core and q2_core:
            q1_set = set(q1_core)
            q2_set = set(q2_core)
            intersection = len(q1_set.intersection(q2_set))
            union = len(q1_set.union(q2_set))
            jaccard = intersection / union if union > 0 else 0
            
            # 5. 计算词序相似度
            # 检查核心词的顺序匹配程度
            order_similarity = 0
            if intersection > 1:  # 至少有2个共同词才考虑顺序
                common_words = q1_set.intersection(q2_set)
                q1_order = [w for w in q1_core if w in common_words]
                q2_order = [w for w in q2_core if w in common_words]
                
                # 计算最长公共子序列
                order_match = 0
                for i in range(min(len(q1_order), len(q2_order))):
                    if q1_order[i] == q2_order[i]:
                        order_match += 1
                        
                max_possible = min(len(q1_order), len(q2_order))
                order_similarity = order_match / max_possible if max_possible > 0 else 0
            
            # 综合评分: Jaccard + 顺序加权
            final_score = jaccard * 0.7 + order_similarity * 0.3
            
            # 特殊情况处理：提升短问题的匹配度
            # 例如"周末去哪"和"周末去哪玩"应该高度相似
            if min(len(q1), len(q2)) < 8 and intersection / min(len(q1_set), len(q2_set)) > 0.7:
                final_score = max(final_score, 0.8)
                
            return final_score
        
        return 0.0 