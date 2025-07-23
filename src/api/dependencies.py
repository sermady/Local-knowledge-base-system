"""API依赖注入"""

import logging
from typing import Annotated
from fastapi import Depends

from src.services import (
    DocumentService,
    HybridSearchService,
    QAService,
    CacheService,
    EmbeddingService,
    VectorService,
    BM25Service
)

logger = logging.getLogger(__name__)

# 全局服务实例
_document_service = None
_search_service = None
_qa_service = None
_cache_service = None
_embedding_service = None
_vector_service = None
_bm25_service = None


async def get_document_service() -> DocumentService:
    """获取文档服务实例"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
        # 初始化服务
        await _document_service.embedding_service.initialize()
        await _document_service.vector_service.initialize()
    return _document_service


async def get_search_service() -> HybridSearchService:
    """获取搜索服务实例"""
    global _search_service
    if _search_service is None:
        _search_service = HybridSearchService()
        await _search_service.initialize()
    return _search_service


async def get_qa_service() -> QAService:
    """获取问答服务实例"""
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
        await _qa_service.initialize()
    return _qa_service


async def get_cache_service() -> CacheService:
    """获取缓存服务实例"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.initialize()
    return _cache_service


async def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        await _embedding_service.initialize()
    return _embedding_service


async def get_vector_service() -> VectorService:
    """获取向量服务实例"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
        await _vector_service.initialize()
    return _vector_service


async def get_bm25_service() -> BM25Service:
    """获取BM25服务实例"""
    global _bm25_service
    if _bm25_service is None:
        _bm25_service = BM25Service()
    return _bm25_service


# 依赖注入类型注解
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
SearchServiceDep = Annotated[HybridSearchService, Depends(get_search_service)]
QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
CacheServiceDep = Annotated[CacheService, Depends(get_cache_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
VectorServiceDep = Annotated[VectorService, Depends(get_vector_service)]
BM25ServiceDep = Annotated[BM25Service, Depends(get_bm25_service)]


async def cleanup_services():
    """清理服务资源"""
    global _document_service, _search_service, _qa_service, _cache_service
    global _embedding_service, _vector_service, _bm25_service
    
    # 这里可以添加服务清理逻辑
    logger.info("清理服务资源")
    
    _document_service = None
    _search_service = None
    _qa_service = None
    _cache_service = None
    _embedding_service = None
    _vector_service = None
    _bm25_service = None
