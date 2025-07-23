"""API路由定义"""

import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query as QueryParam
from pydantic import BaseModel

from src.models.document import DocumentInfo
from src.models.search import QAResponse
from src.models.query import Query, QueryResult
from src.models.cache import CacheStats
from src.api.dependencies import (
    DocumentServiceDep,
    SearchServiceDep,
    QAServiceDep,
    CacheServiceDep
)
from src.utils.performance import performance_monitor

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


# 请求/响应模型
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    document_ids: Optional[List[str]] = None


class QARequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document_info: DocumentInfo
    message: str


# 文档管理API
@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = None,
    document_service: DocumentServiceDep = None
):
    """上传文档"""
    try:
        # 解析元数据
        import json
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"无效的元数据格式: {metadata}")
        
        # 上传文档
        doc_info = await document_service.upload_document(file, metadata_dict)
        
        return DocumentUploadResponse(
            document_info=doc_info,
            message="文档上传成功，正在后台处理中"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail="文档上传失败")


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    limit: int = QueryParam(50, ge=1, le=100),
    offset: int = QueryParam(0, ge=0),
    document_service: DocumentServiceDep = None
):
    """列出文档"""
    try:
        documents = await document_service.list_documents(limit, offset)
        return documents
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取文档列表失败")


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str, document_service: DocumentServiceDep = None):
    """获取文档信息"""
    try:
        doc_info = await document_service.get_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取文档信息失败")


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    document_service: DocumentServiceDep = None,
    cache_service: CacheServiceDep = None
):
    """删除文档"""
    try:
        success = await document_service.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="文档不存在或删除失败")
        
        # 使相关缓存失效
        await cache_service.invalidate_cache(doc_id)
        
        return {"message": "文档删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail="删除文档失败")


# 搜索API
@router.post("/search", response_model=QueryResult)
async def search_documents(request: SearchRequest, search_service: SearchServiceDep = None):
    """搜索文档"""
    try:
        # 创建查询对象
        query = Query(text=request.query)
        
        # 执行混合搜索
        result = await search_service.search(
            query=query,
            document_ids=request.document_ids
        )
        
        # 限制结果数量
        if len(result.chunks) > request.limit:
            result.chunks = result.chunks[:request.limit]
            result.total_results = len(result.chunks)
        
        return result
        
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail="搜索失败")


@router.post("/search/vector", response_model=QueryResult)
async def vector_search(request: SearchRequest, search_service: SearchServiceDep = None):
    """仅使用向量搜索"""
    try:
        query = Query(text=request.query)
        result = await search_service.search_vector_only(
            query=query,
            limit=request.limit,
            document_ids=request.document_ids
        )
        
        return result
        
    except Exception as e:
        logger.error(f"向量搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail="向量搜索失败")


@router.post("/search/bm25", response_model=QueryResult)
async def bm25_search(request: SearchRequest, search_service: SearchServiceDep = None):
    """仅使用BM25搜索"""
    try:
        query = Query(text=request.query)
        result = await search_service.search_bm25_only(
            query=query,
            limit=request.limit,
            document_ids=request.document_ids
        )
        
        return result
        
    except Exception as e:
        logger.error(f"BM25搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail="BM25搜索失败")


# 问答API
@router.post("/qa", response_model=QAResponse)
async def ask_question(
    request: QARequest,
    search_service: SearchServiceDep = None,
    qa_service: QAServiceDep = None,
    cache_service: CacheServiceDep = None
):
    """问答接口"""
    try:
        # 检查缓存
        cached_result = await cache_service.get_cached_result(
            request.question, 
            request.document_ids
        )
        
        if cached_result:
            return cached_result
        
        # 搜索相关文档
        search_query = Query(text=request.question)
        search_result = await search_service.search(
            query=search_query,
            document_ids=request.document_ids
        )
        
        # 提取文本块 - 暂时使用空列表，后续需要完善
        context_chunks = []
        # TODO: 将搜索结果转换为TextChunk对象
        
        # 生成答案
        qa_response = await qa_service.generate_answer(
            question=request.question,
            context_chunks=context_chunks,
            conversation_id=request.conversation_id
        )
        
        # 缓存结果
        await cache_service.cache_result(
            request.question,
            qa_response,
            request.document_ids
        )
        
        return qa_response
        
    except Exception as e:
        logger.error(f"问答处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail="问答处理失败")


# 系统状态API
@router.get("/system/status")
async def get_system_status(
    search_service: SearchServiceDep = None,
    cache_service: CacheServiceDep = None
):
    """获取系统状态"""
    try:
        # 获取各服务状态
        search_stats = await search_service.get_search_stats()
        cache_stats = await cache_service.get_cache_stats()
        
        # 获取性能指标
        performance_stats = performance_monitor.get_metrics()

        return {
            "status": "healthy",
            "services": {
                "search": search_stats,
                "cache": cache_stats.model_dump()
            },
            "performance": performance_stats
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取系统状态失败")


@router.get("/system/cache/stats", response_model=CacheStats)
async def get_cache_stats(cache_service: CacheServiceDep = None):
    """获取缓存统计"""
    try:
        return await cache_service.get_cache_stats()
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取缓存统计失败")


@router.post("/system/cache/cleanup")
async def cleanup_cache(cache_service: CacheServiceDep = None):
    """清理过期缓存"""
    try:
        cleaned_count = await cache_service.cleanup_expired_cache()
        return {
            "message": f"清理了 {cleaned_count} 个过期缓存条目"
        }
        
    except Exception as e:
        logger.error(f"清理缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail="清理缓存失败")


@router.get("/system/performance")
async def get_performance_metrics():
    """获取性能指标"""
    try:
        return performance_monitor.get_metrics()

    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取性能指标失败")


@router.post("/system/performance/reset")
async def reset_performance_metrics():
    """重置性能指标"""
    try:
        performance_monitor.reset_metrics()
        return {"message": "性能指标已重置"}

    except Exception as e:
        logger.error(f"重置性能指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail="重置性能指标失败")


# 健康检查API
@router.get("/health")
async def health_check():
    """健康检查"""
    import time
    return {
        "status": "healthy",
        "timestamp": time.time()
    }
