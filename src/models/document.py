"""文档相关数据模型"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """文档状态枚举"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentInfo(BaseModel):
    """文档基本信息"""
    id: UUID = Field(default_factory=uuid4)
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    status: DocumentStatus = DocumentStatus.UPLOADING
    user_id: Optional[UUID] = None
    metadata: Dict = Field(default_factory=dict)


class Document(BaseModel):
    """完整文档模型"""
    info: DocumentInfo
    content: Optional[str] = None
    parsed_content: Optional[Dict] = None
    chunk_count: int = 0
    entity_count: int = 0
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
