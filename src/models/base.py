"""
基础数据模型定义
Base data model definitions
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ProcessingStatus(str, Enum):
    """文档处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EntityType(str, Enum):
    """实体类型枚举"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PRODUCT = "product"
    EVENT = "event"
    OTHER = "other"


class BaseDataModel(BaseModel):
    """基础数据模型，提供通用字段和方法"""
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="唯一标识符")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="创建时间")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="更新时间")
        
    def update_timestamp(self) -> None:
        """更新时间戳"""
        self.updated_at = datetime.now(timezone.utc)
        
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """验证ID格式"""
        if not v or len(v.strip()) == 0:
            raise ValueError("ID不能为空")
        return v.strip()