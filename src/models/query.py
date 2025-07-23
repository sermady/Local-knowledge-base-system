"""查询相关数据模型"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field, field_validator

from .base import BaseDataModel
from .search import QAResponse


class Query(BaseDataModel):
    """查询数据模型"""
    
    text: str = Field(..., description="查询文本")
    query_type: str = Field(default="hybrid", description="查询类型")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    limit: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相关性阈值")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """验证查询文本"""
        if not v or len(v.strip()) == 0:
            raise ValueError("查询文本不能为空")
        return v.strip()
    
    @field_validator('query_type')
    @classmethod
    def validate_query_type(cls, v: str) -> str:
        """验证查询类型"""
        valid_types = {'vector', 'bm25', 'hybrid'}
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


class QueryResult(BaseDataModel):
    """查询结果数据模型"""
    
    query_id: str = Field(..., description="查询ID")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="结果块列表")
    total_results: int = Field(default=0, ge=0, description="总结果数")
    search_time: float = Field(default=0.0, ge=0.0, description="搜索时间（秒）")
    retrieval_method: str = Field(default="hybrid", description="检索方法")
    used_cache: bool = Field(default=False, description="是否使用缓存")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @field_validator('retrieval_method')
    @classmethod
    def validate_retrieval_method(cls, v: str) -> str:
        """验证检索方法"""
        valid_methods = {'vector', 'bm25', 'hybrid'}
        if v not in valid_methods:
            raise ValueError(f"无效的检索方法: {v}")
        return v
    
    def get_result_count(self) -> int:
        """获取结果数量"""
        return len(self.chunks)
    
    def get_top_result(self) -> Optional[Dict[str, Any]]:
        """获取最佳结果"""
        return self.chunks[0] if self.chunks else None
    
    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """添加结果块"""
        self.chunks.append(chunk)
        self.total_results = len(self.chunks)
        self.update_timestamp()
    
    def sort_by_score(self, reverse: bool = True) -> None:
        """按分数排序"""
        self.chunks.sort(key=lambda x: x.get('score', 0.0), reverse=reverse)
        self.update_timestamp()


class QueryHistory(BaseDataModel):
    """查询历史数据模型"""
    
    user_id: str = Field(..., description="用户ID")
    queries: List[Query] = Field(default_factory=list, description="查询列表")
    session_id: Optional[str] = Field(None, description="会话ID")
    
    def add_query(self, query: Query) -> None:
        """添加查询"""
        self.queries.append(query)
        self.update_timestamp()
    
    def get_recent_queries(self, limit: int = 10) -> List[Query]:
        """获取最近的查询"""
        return self.queries[-limit:] if limit > 0 else self.queries
    
    def get_query_count(self) -> int:
        """获取查询数量"""
        return len(self.queries)
    
    def clear_history(self) -> None:
        """清空历史"""
        self.queries.clear()
        self.update_timestamp()


class QueryAnalytics(BaseDataModel):
    """查询分析数据模型"""
    
    query_id: str = Field(..., description="查询ID")
    query_text: str = Field(..., description="查询文本")
    result_count: int = Field(default=0, ge=0, description="结果数量")
    search_time: float = Field(default=0.0, ge=0.0, description="搜索时间")
    user_id: Optional[str] = Field(None, description="用户ID")
    clicked_results: List[str] = Field(default_factory=list, description="点击的结果ID列表")
    user_feedback: Optional[str] = Field(None, description="用户反馈")
    satisfaction_score: Optional[float] = Field(None, ge=0.0, le=5.0, description="满意度评分")
    
    def add_click(self, result_id: str) -> None:
        """添加点击记录"""
        if result_id not in self.clicked_results:
            self.clicked_results.append(result_id)
            self.update_timestamp()
    
    def set_feedback(self, feedback: str, score: Optional[float] = None) -> None:
        """设置用户反馈"""
        self.user_feedback = feedback.strip() if feedback else None
        if score is not None:
            self.satisfaction_score = max(0.0, min(5.0, score))
        self.update_timestamp()
    
    def get_click_count(self) -> int:
        """获取点击数量"""
        return len(self.clicked_results)


class QuerySuggestion(BaseDataModel):
    """查询建议数据模型"""
    
    original_query: str = Field(..., description="原始查询")
    suggested_query: str = Field(..., description="建议查询")
    suggestion_type: str = Field(..., description="建议类型")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度")
    reason: Optional[str] = Field(None, description="建议原因")
    
    @field_validator('suggestion_type')
    @classmethod
    def validate_suggestion_type(cls, v: str) -> str:
        """验证建议类型"""
        valid_types = {'spelling', 'synonym', 'expansion', 'refinement'}
        if v not in valid_types:
            raise ValueError(f"无效的建议类型: {v}")
        return v
    
    @field_validator('original_query', 'suggested_query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """验证查询文本"""
        if not v or len(v.strip()) == 0:
            raise ValueError("查询文本不能为空")
        return v.strip()
