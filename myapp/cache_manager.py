import os
import time
import logging
import threading
from typing import Any, Dict, Optional
import torch
from .config import config

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_times: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.max_size = 1000  # 最大缓存项数
        self.expire_time = 3600  # 缓存过期时间（秒）
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 检查是否过期
                if time.time() - self.cache_times[key] > self.expire_time:
                    self._remove(key)
                    return None
                return self.cache[key]
            return None
            
    def set(self, key: str, value: Any):
        """设置缓存项"""
        with self.lock:
            # 检查缓存大小
            if len(self.cache) >= self.max_size:
                self._remove_oldest()
            
            self.cache[key] = value
            self.cache_times[key] = time.time()
            
    def _remove(self, key: str):
        """移除缓存项"""
        if key in self.cache:
            del self.cache[key]
            del self.cache_times[key]
            
    def _remove_oldest(self):
        """移除最旧的缓存项"""
        if not self.cache:
            return
            
        oldest_key = min(self.cache_times.items(), key=lambda x: x[1])[0]
        self._remove(oldest_key)
        
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.cache_times.clear()

class ModelCache:
    def __init__(self):
        self.cache_manager = CacheManager()
        self.model = None
        self.tokenizer = None
        self.last_used = None
        self.lock = threading.Lock()
        
    def get_model(self):
        """获取模型实例"""
        with self.lock:
            if self.model is None:
                self._load_model()
            self.last_used = time.time()
            return self.model, self.tokenizer
            
    def _load_model(self):
        """加载模型"""
        try:
            model_path = config.get_model_config()['MODEL_PATH']
            model_name = config.get_model_config()['MODEL_NAME']
            
            # 检查缓存
            cached_model = self.cache_manager.get(f"model_{model_name}")
            if cached_model:
                self.model, self.tokenizer = cached_model
                return
                
            # 加载模型
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            
            # 设置设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # 缓存模型
            self.cache_manager.set(f"model_{model_name}", (self.model, self.tokenizer))
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

class DocumentCache:
    def __init__(self):
        self.cache_manager = CacheManager()
        
    def get_document(self, doc_id: str) -> Optional[str]:
        """获取文档内容"""
        return self.cache_manager.get(f"doc_{doc_id}")
        
    def set_document(self, doc_id: str, content: str):
        """缓存文档内容"""
        self.cache_manager.set(f"doc_{doc_id}", content)
        
    def get_embedding(self, doc_id: str) -> Optional[torch.Tensor]:
        """获取文档嵌入向量"""
        return self.cache_manager.get(f"emb_{doc_id}")
        
    def set_embedding(self, doc_id: str, embedding: torch.Tensor):
        """缓存文档嵌入向量"""
        self.cache_manager.set(f"emb_{doc_id}", embedding)

# 创建全局缓存实例
model_cache = ModelCache()
document_cache = DocumentCache() 