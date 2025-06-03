import os
import json
import logging
from pathlib import Path

class Config:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.load_config()
        
    def load_config(self):
        """加载配置"""
        # 模型配置
        self.MODEL_CONFIG = {
            'MODEL_NAME': 'bert-base-chinese',
            'MODEL_PATH': os.path.join(self.BASE_DIR, 'models', 'bert_qa'),
            'KB_DIR': os.path.join(self.BASE_DIR, 'knowledge_base'),
            'TEMP_MODEL_PATH': os.path.join(self.BASE_DIR, 'models', 'temp'),
            'MAX_LENGTH': 384,
            'TOP_K': 3,
            'THRESHOLD': 0.5,
            'BATCH_SIZE': 16,
            'LEARNING_RATE': 2e-5,
            'EPOCHS': 3
        }
        
        # 系统配置
        self.SYSTEM_CONFIG = {
            'LOG_DIR': os.path.join(self.BASE_DIR, 'logs'),
            'CACHE_DIR': os.path.join(self.BASE_DIR, 'cache'),
            'MAX_DOCUMENTS': 10000,
            'CHUNK_SIZE': 500,
            'CHUNK_OVERLAP': 100,
            'MAX_MEMORY_USAGE': 0.8,  # 最大内存使用比例
            'REQUEST_TIMEOUT': 30,     # 请求超时时间（秒）
            'RATE_LIMIT': 100          # 每分钟最大请求数
        }
        
        # 安全配置
        self.SECURITY_CONFIG = {
            'ALLOWED_HOSTS': ['localhost', '127.0.0.1'],
            'SECRET_KEY': os.environ.get('DJANGO_SECRET_KEY', 'your-secret-key'),
            'DEBUG': False,
            'ENCRYPTION_KEY': os.environ.get('ENCRYPTION_KEY', 'your-encryption-key')
        }
        
        # 创建必要的目录
        self._create_directories()
        
        # 配置日志
        self._setup_logging()
        
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.MODEL_CONFIG['MODEL_PATH'],
            self.MODEL_CONFIG['KB_DIR'],
            self.MODEL_CONFIG['TEMP_MODEL_PATH'],
            self.SYSTEM_CONFIG['LOG_DIR'],
            self.SYSTEM_CONFIG['CACHE_DIR']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def _setup_logging(self):
        """配置日志系统"""
        log_file = os.path.join(self.SYSTEM_CONFIG['LOG_DIR'], 'app.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def get_model_config(self):
        """获取模型配置"""
        return self.MODEL_CONFIG
        
    def get_system_config(self):
        """获取系统配置"""
        return self.SYSTEM_CONFIG
        
    def get_security_config(self):
        """获取安全配置"""
        return self.SECURITY_CONFIG

# 创建全局配置实例
config = Config() 