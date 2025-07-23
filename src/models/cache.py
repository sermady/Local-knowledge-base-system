"""
缓存相关数据模型
Cache-related data models
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from pydantic import Field, field_validator

from .base import BaseDataModel
from .search import QAResponse


class CacheEntry(BaseDataModel):
    """缓存条目数据模型"""
    
    key: str = Field(..., description="缓存键")
    value: Dict[str, Any] = Field(..., description="缓存值")
    ttl: int = Field(..., gt=0, description="生存时间（秒）")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="创建时间")
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="最后访问时间")
    access_count: int = Field(default=0, ge=0, description="访问次数")
    size_bytes: int = Field(default=0, ge=0, description="大小（字节）")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v: str) -> str:
        """验证缓存键"""
        if not v or len(v.strip()) == 0:
            raise ValueError("缓存键不能为空")
        return v.strip()
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        expiry_time = self.created_at + timedelta(seconds=self.ttl)
        return datetime.now(timezone.utc) > expiry_time
    
    def get_remaining_ttl(self) -> int:
        """获取剩余生存时间（秒）"""
        if self.is_expired():
            return 0
        expiry_time = self.created_at + timedelta(seconds=self.ttl)
        remaining = expiry_time - datetime.now(timezone.utc)
        return max(0, int(remaining.total_seconds()))
    
    def update_access(self) -> None:
        """更新访问信息"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        self.update_timestamp()
    
    def extend_ttl(self, additional_seconds: int) -> None:
        """延长生存时间"""
        if additional_seconds > 0:
            self.ttl += additional_seconds
            self.update_timestamp()


class QueryCache(BaseDataModel):
    """查询缓存数据模型"""
    
    query_hash: str = Field(..., description="查询哈希值")
    query_text: str = Field(..., description="查询文本")
    response: QAResponse = Field(..., description="问答响应")
    document_ids: list[str] = Field(default_factory=list, description="相关文档ID列表")
    ttl: int = Field(default=3600, gt=0, description="生存时间（秒）")
    hit_count: int = Field(default=0, ge=0, description="命中次数")
    
    @field_validator('query_hash')
    @classmethod
    def validate_query_hash(cls, v: str) -> str:
        """验证查询哈希值"""
        if not v or len(v.strip()) == 0:
            raise ValueError("查询哈希值不能为空")
        # 简单验证哈希值格式
        if len(v) != 64:  # SHA-256
            raise ValueError("无效的哈希值格式")
        return v.strip()
    
    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """验证查询文本"""
        if not v or len(v.strip()) == 0:
            raise ValueError("查询文本不能为空")
        return v.strip()
    
    def increment_hit_count(self) -> None:
        """增加命中次数"""
        self.hit_count += 1
        self.update_timestamp()
    
    def is_valid_for_documents(self, current_document_ids: list[str]) -> bool:
        """检查对于当前文档集合是否有效"""
        return set(self.document_ids) == set(current_document_ids)


class CacheStats(BaseDataModel):
    """缓存统计数据模型"""
    
    total_entries: int = Field(default=0, ge=0, description="总条目数")
    total_size_bytes: int = Field(default=0, ge=0, description="总大小（字节）")
    hit_count: int = Field(default=0, ge=0, description="命中次数")
    miss_count: int = Field(default=0, ge=0, description="未命中次数")
    eviction_count: int = Field(default=0, ge=0, description="驱逐次数")
    expired_count: int = Field(default=0, ge=0, description="过期次数")
    average_access_time: float = Field(default=0.0, ge=0.0, description="平均访问时间（毫秒）")
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        return self.hit_count / total_requests
    
    def get_miss_rate(self) -> float:
        """获取未命中率"""
        return 1.0 - self.get_hit_rate()
    
    def record_hit(self) -> None:
        """记录命中"""
        self.hit_count += 1
        self.update_timestamp()
    
    def record_miss(self) -> None:
        """记录未命中"""
        self.miss_count += 1
        self.update_timestamp()
    
    def record_eviction(self) -> None:
        """记录驱逐"""
        self.eviction_count += 1
        self.update_timestamp()
    
    def record_expiration(self) -> None:
        """记录过期"""
        self.expired_count += 1
        self.update_timestamp()
    
    def update_size(self, size_change: int) -> None:
        """更新大小"""
        self.total_size_bytes = max(0, self.total_size_bytes + size_change)
        self.update_timestamp()
    
    def update_entry_count(self, count_change: int) -> None:
        """更新条目数"""
        self.total_entries = max(0, self.total_entries + count_change)
        self.update_timestamp()


class CacheConfig(BaseDataModel):
    """缓存配置数据模型"""
    
    max_size_bytes: int = Field(default=100 * 1024 * 1024, gt=0, description="最大大小（字节）")
    max_entries: int = Field(default=10000, gt=0, description="最大条目数")
    default_ttl: int = Field(default=3600, gt=0, description="默认生存时间（秒）")
    cleanup_interval: int = Field(default=300, gt=0, description="清理间隔（秒）")
    eviction_policy: str = Field(default="lru", description="驱逐策略")
    enable_compression: bool = Field(default=True, description="是否启用压缩")
    
    @field_validator('eviction_policy')
    @classmethod
    def validate_eviction_policy(cls, v: str) -> str:
        """验证驱逐策略"""
        valid_policies = {'lru', 'lfu', 'fifo', 'random'}
        if v.lower() not in valid_policies:
            raise ValueError(f"无效的驱逐策略: {v}")
        return v.lower()
    
    def get_max_size_mb(self) -> float:
        """获取最大大小（MB）"""
        return self.max_size_bytes / (1024 * 1024)
    
    def set_max_size_mb(self, size_mb: float) -> None:
        """设置最大大小（MB）"""
        self.max_size_bytes = int(size_mb * 1024 * 1024)
        self.update_timestamp()


class CacheOperation(BaseDataModel):
    """缓存操作记录数据模型"""
    
    operation_type: str = Field(..., description="操作类型")
    key: str = Field(..., description="缓存键")
    success: bool = Field(..., description="是否成功")
    execution_time: float = Field(..., ge=0.0, description="执行时间（毫秒）")
    error_message: Optional[str] = Field(None, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @field_validator('operation_type')
    @classmethod
    def validate_operation_type(cls, v: str) -> str:
        """验证操作类型"""
        valid_operations = {
            'get', 'set', 'delete', 'clear', 'expire', 
            'cleanup', 'evict', 'stats'
        }
        if v.lower() not in valid_operations:
            raise ValueError(f"无效的操作类型: {v}")
        return v.lower()
    
    def is_read_operation(self) -> bool:
        """是否为读操作"""
        return self.operation_type in {'get', 'stats'}
    
    def is_write_operation(self) -> bool:
        """是否为写操作"""
        return self.operation_type in {'set', 'delete', 'clear', 'expire'}
    
    def is_maintenance_operation(self) -> bool:
        """是否为维护操作"""
        return self.operation_type in {'cleanup', 'evict'}