"""
搜索相关数据模型
Search-related data models
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import BaseDataModel


class SearchResult(BaseDataModel):
    """搜索结果数据模型"""
    
    chunk_id: str = Field(..., description="文本块ID")
    document_id: str = Field(..., description="文档ID")
    content: str = Field(..., description="文本内容")
    score: float = Field(..., ge=0.0, description="相关性评分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    highlight: Optional[str] = Field(None, description="高亮文本")
    document_title: Optional[str] = Field(None, description="文档标题")
    page_number: Optional[int] = Field(None, ge=1, description="页码")
    section_title: Optional[str] = Field(None, description="章节标题")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """验证内容"""
        if not v or len(v.strip()) == 0:
            raise ValueError("搜索结果内容不能为空")
        return v.strip()
    
    @field_validator('chunk_id')
    @classmethod
    def validate_chunk_id(cls, v: str) -> str:
        """验证文本块ID"""
        if not v or len(v.strip()) == 0:
            raise ValueError("文本块ID不能为空")
        return v.strip()
    
    @field_validator('document_id')
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """验证文档ID"""
        if not v or len(v.strip()) == 0:
            raise ValueError("文档ID不能为空")
        return v.strip()
    
    def get_content_preview(self, max_length: int = 200) -> str:
        """获取内容预览"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def has_highlight(self) -> bool:
        """是否有高亮文本"""
        return self.highlight is not None and len(self.highlight.strip()) > 0


class SearchQuery(BaseDataModel):
    """搜索查询数据模型"""
    
    query_text: str = Field(..., description="查询文本")
    query_type: str = Field(default="hybrid", description="查询类型")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相关性阈值")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    highlight_query: bool = Field(default=True, description="是否高亮查询词")
    
    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """验证查询文本"""
        if not v or len(v.strip()) == 0:
            raise ValueError("查询文本不能为空")
        return v.strip()
    
    @field_validator('query_type')
    @classmethod
    def validate_query_type(cls, v: str) -> str:
        """验证查询类型"""
        valid_types = {'vector', 'keyword', 'hybrid'}
        if v not in valid_types:
            raise ValueError(f"无效的查询类型: {v}")
        return v
    
    def add_filter(self, key: str, value: Any) -> None:
        """添加过滤条件"""
        self.filters[key] = value
        self.update_timestamp()
    
    def remove_filter(self, key: str) -> None:
        """移除过滤条件"""
        if key in self.filters:
            del self.filters[key]
            self.update_timestamp()


class SearchResponse(BaseDataModel):
    """搜索响应数据模型"""
    
    query: SearchQuery = Field(..., description="原始查询")
    results: List[SearchResult] = Field(default_factory=list, description="搜索结果")
    total_count: int = Field(default=0, ge=0, description="总结果数")
    processing_time: float = Field(default=0.0, ge=0.0, description="处理时间（秒）")
    used_cache: bool = Field(default=False, description="是否使用缓存")
    search_strategy: str = Field(default="hybrid", description="使用的搜索策略")
    
    def get_result_count(self) -> int:
        """获取结果数量"""
        return len(self.results)
    
    def get_top_result(self) -> Optional[SearchResult]:
        """获取最佳结果"""
        return self.results[0] if self.results else None
    
    def get_results_by_document(self, document_id: str) -> List[SearchResult]:
        """按文档ID获取结果"""
        return [result for result in self.results if result.document_id == document_id]
    
    def filter_by_score(self, min_score: float) -> List[SearchResult]:
        """按评分过滤结果"""
        return [result for result in self.results if result.score >= min_score]


class Citation(BaseDataModel):
    """引用数据模型"""
    
    document_id: str = Field(..., description="文档ID")
    document_title: str = Field(..., description="文档标题")
    chunk_id: str = Field(..., description="文本块ID")
    page_number: Optional[int] = Field(None, ge=1, description="页码")
    section_title: Optional[str] = Field(None, description="章节标题")
    quoted_text: str = Field(..., description="引用文本")
    context: Optional[str] = Field(None, description="上下文")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="相关性评分")
    
    @field_validator('document_title')
    @classmethod
    def validate_document_title(cls, v: str) -> str:
        """验证文档标题"""
        if not v or len(v.strip()) == 0:
            raise ValueError("文档标题不能为空")
        return v.strip()
    
    @field_validator('quoted_text')
    @classmethod
    def validate_quoted_text(cls, v: str) -> str:
        """验证引用文本"""
        if not v or len(v.strip()) == 0:
            raise ValueError("引用文本不能为空")
        return v.strip()
    
    def get_citation_format(self) -> str:
        """获取格式化的引用"""
        citation = f"《{self.document_title}》"
        if self.page_number:
            citation += f", 第{self.page_number}页"
        if self.section_title:
            citation += f", {self.section_title}"
        return citation
    
    def get_quoted_preview(self, max_length: int = 100) -> str:
        """获取引用文本预览"""
        if len(self.quoted_text) <= max_length:
            return self.quoted_text
        return self.quoted_text[:max_length] + "..."


class QAResponse(BaseDataModel):
    """问答响应数据模型"""
    
    question: str = Field(..., description="用户问题")
    answer: str = Field(..., description="回答内容")
    sources: List[Citation] = Field(default_factory=list, description="来源引用")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度")
    processing_time: float = Field(default=0.0, ge=0.0, description="处理时间（秒）")
    cached: bool = Field(default=False, description="是否来自缓存")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    has_context: bool = Field(default=True, description="是否基于文档内容")
    api_usage: Dict[str, Any] = Field(default_factory=dict, description="API使用情况")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        """验证问题"""
        if not v or len(v.strip()) == 0:
            raise ValueError("问题不能为空")
        return v.strip()
    
    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """验证回答"""
        if not v or len(v.strip()) == 0:
            raise ValueError("回答不能为空")
        return v.strip()
    
    def get_source_count(self) -> int:
        """获取来源数量"""
        return len(self.sources)
    
    def get_primary_source(self) -> Optional[Citation]:
        """获取主要来源"""
        if not self.sources:
            return None
        return max(self.sources, key=lambda x: x.relevance_score)
    
    def add_source(self, citation: Citation) -> None:
        """添加来源"""
        self.sources.append(citation)
        self.update_timestamp()
    
    def is_no_context_response(self) -> bool:
        """是否为无相关文档的回答"""
        return not self.has_context or len(self.sources) == 0
    
    def get_answer_preview(self, max_length: int = 200) -> str:
        """获取回答预览"""
        if len(self.answer) <= max_length:
            return self.answer
        return self.answer[:max_length] + "..."