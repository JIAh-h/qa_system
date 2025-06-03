import time
import logging
import hashlib
import hmac
from typing import Dict, List, Optional
from django.http import HttpRequest
from .config import config

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.rate_limit = config.get_system_config()['RATE_LIMIT']
        self.window = 60  # 时间窗口（秒）
        
    def check(self, request: HttpRequest) -> bool:
        """检查请求是否超过频率限制"""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # 清理过期的请求记录
        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip]
                if current_time - t < self.window
            ]
            
        # 检查请求频率
        if client_ip not in self.requests:
            self.requests[client_ip] = []
            
        if len(self.requests[client_ip]) >= self.rate_limit:
            return False
            
        self.requests[client_ip].append(current_time)
        return True
        
    def _get_client_ip(self, request: HttpRequest) -> str:
        """获取客户端IP"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR', 'unknown')

class DataEncryption:
    def __init__(self):
        self.key = config.get_security_config()['ENCRYPTION_KEY'].encode()
        
    def encrypt(self, data: str) -> str:
        """加密数据"""
        try:
            # 使用HMAC-SHA256进行加密
            h = hmac.new(self.key, data.encode(), hashlib.sha256)
            return h.hexdigest()
        except Exception as e:
            logger.error(f"加密数据失败: {str(e)}")
            return ""
            
    def verify(self, data: str, encrypted: str) -> bool:
        """验证数据"""
        try:
            return self.encrypt(data) == encrypted
        except Exception as e:
            logger.error(f"验证数据失败: {str(e)}")
            return False

class AccessControl:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.encryption = DataEncryption()
        self.allowed_hosts = config.get_security_config()['ALLOWED_HOSTS']
        
    def check_access(self, request: HttpRequest) -> tuple[bool, str]:
        """检查访问权限"""
        # 检查主机
        if not self._check_host(request):
            return False, "主机未授权"
            
        # 检查频率限制
        if not self.rate_limiter.check(request):
            return False, "请求频率超限"
            
        return True, "访问通过"
        
    def _check_host(self, request: HttpRequest) -> bool:
        """检查主机是否允许访问"""
        host = request.get_host().split(':')[0]
        return host in self.allowed_hosts
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        return self.encryption.encrypt(data)
        
    def verify_data(self, data: str, encrypted: str) -> bool:
        """验证数据完整性"""
        return self.encryption.verify(data, encrypted)

# 创建全局安全控制实例
access_control = AccessControl() 