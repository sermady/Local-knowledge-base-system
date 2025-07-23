"""
文本块相关数据模型
Text chunk related data models
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import BaseDataModel


class TextChunk(BaseDataModel):
    """文本块数据模型"""
    
    document_id: str = Field(..., description="所属文档ID")
    content: str = Field(..., description="文本内容")
    chunk_index: int = Field(..., ge=0, description="块索引")
    start_position: int = Field(..., ge=0, description="在原文档中的起始位置")
    end_position: int = Field(..., ge=0, description="在原文档中的结束位置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    embedding: Optional[List[float]] = Field(None, description="向量表示")
    token_count: Optional[int] = Field(None, ge=0, description="Token数量")
    language: str = Field(default="zh", description="语言")
    page_number: Optional[int] = Field(None, ge=1, description="页码")
    section_title: Optional[str] = Field(None, description="章节标题")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """验证文本内容"""
        if not v or len(v.strip()) == 0:
            raise ValueError("文本内容不能为空")
        # 检查内容长度（避免过长的文本块）
        if len(v) > 10000:
            raise ValueError("文本块长度不能超过10000个字符")
        return v.strip()
    
    @field_validator('end_position')
    @classmethod
    def validate_positions(cls, v: int, info) -> int:
        """验证位置信息"""
        if info.data:
            start_pos = info.data.get('start_position', 0)
            if v <= start_pos:
                raise ValueError("结束位置必须大于起始位置")
        return v
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """验证向量表示"""
        if v is not None:
            if len(v) == 0:
                raise ValueError("向量不能为空列表")
            # 检查向量维度（假设使用768维的向量）
            if len(v) not in [384, 512, 768, 1024, 1536]:
                raise ValueError(f"不支持的向量维度: {len(v)}")
        return v
    
    def get_content_length(self) -> int:
        """获取内容长度"""
        return len(self.content)
    
    def get_position_range(self) -> tuple[int, int]:
        """获取位置范围"""
        return (self.start_position, self.end_position)
    
    def has_embedding(self) -> bool:
        """是否有向量表示"""
        return self.embedding is not None and len(self.embedding) > 0
    
    def set_embedding(self, embedding: List[float]) -> None:
        """设置向量表示"""
        if not embedding:
            raise ValueError("向量不能为空")
        self.embedding = embedding
        self.update_timestamp()
    
    def calculate_similarity(self, other_embedding: List[float]) -> float:
        """计算与另一个向量的相似度（余弦相似度）"""
        if not self.has_embedding():
            raise ValueError("当前文本块没有向量表示")
        if not other_embedding:
            raise ValueError("比较向量不能为空")
        if len(self.embedding) != len(other_embedding):
            raise ValueError("向量维度不匹配")
        
        # 计算余弦相似度
        import math
        
        dot_product = sum(a * b for a, b in zip(self.embedding, other_embedding))
        norm_a = math.sqrt(sum(a * a for a in self.embedding))
        norm_b = math.sqrt(sum(b * b for b in other_embedding))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


class ChunkMetadata(BaseDataModel):
    """文本块元数据"""
    
    chunk_id: str = Field(..., description="文本块ID")
    document_title: Optional[str] = Field(None, description="文档标题")
    section_hierarchy: List[str] = Field(
        default_factory=list, 
        description="章节层次结构"
    )
    keywords: List[str] = Field(default_factory=list, description="关键词")
    entities: List[str] = Field(default_factory=list, description="实体列表")
    sentiment: Optional[str] = Field(None, description="情感倾向")
    importance_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="重要性评分"
    )
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """验证关键词"""
        return list(set(kw.strip() for kw in v if kw.strip()))
    
    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v: List[str]) -> List[str]:
        """验证实体列表"""
        return list(set(entity.strip() for entity in v if entity.strip()))
    
    def add_keyword(self, keyword: str) -> None:
        """添加关键词"""
        keyword = keyword.strip()
        if keyword and keyword not in self.keywords:
            self.keywords.append(keyword)
            self.update_timestamp()
    
    def add_entity(self, entity: str) -> None:
        """添加实体"""
        entity = entity.strip()
        if entity and entity not in self.entities:
            self.entities.append(entity)
            self.update_timestamp()


class Vector(BaseDataModel):
    """向量数据模型"""
    
    chunk_id: str = Field(..., description="文本块ID")
    embedding: List[float] = Field(..., description="向量表示")
    dimension: int = Field(..., ge=1, description="向量维度")
    model_name: str = Field(..., description="生成向量的模型名称")
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """验证向量"""
        if not v:
            raise ValueError("向量不能为空")
        return v
    
    @field_validator('dimension')
    @classmethod
    def validate_dimension(cls, v: int, info) -> int:
        """验证向量维度"""
        if info.data:
            embedding = info.data.get('embedding', [])
            if embedding and len(embedding) != v:
                raise ValueError("向量维度与实际长度不匹配")
        return v
    
    def get_norm(self) -> float:
        """计算向量的模长"""
        import math
        return math.sqrt(sum(x * x for x in self.embedding))
    
    def normalize(self) -> List[float]:
        """归一化向量"""
        norm = self.get_norm()
        if norm == 0:
            return self.embedding
        return [x / norm for x in self.embedding]