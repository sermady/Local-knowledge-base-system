"""混合检索服务"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.services.vector_service import VectorService
from src.services.bm25_service import BM25Service
from src.services.embedding_service import EmbeddingService
from src.models.query import Query, QueryResult
from src.config.settings import get_settings
from src.utils.performance import monitor_performance, measure_time

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SearchConfig:
    """搜索配置"""
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    min_score_threshold: float = 0.1
    max_results: int = 20
    enable_rerank: bool = True


class HybridSearchService:
    """混合检索服务"""
    
    def __init__(self):
        self.vector_service = VectorService()
        self.bm25_service = BM25Service()
        self.embedding_service = EmbeddingService()
        self.default_config = SearchConfig()
    
    async def initialize(self):
        """初始化所有服务"""
        await asyncio.gather(
            self.vector_service.initialize(),
            self.embedding_service.initialize()
        )
        logger.info("混合检索服务初始化完成")
    
    @monitor_performance("hybrid_search")
    async def search(
        self,
        query: Query,
        config: Optional[SearchConfig] = None,
        document_ids: Optional[List[str]] = None
    ) -> QueryResult:
        """执行混合检索"""
        if config is None:
            config = self.default_config
        
        try:
            start_time = asyncio.get_event_loop().time()

            # 并行执行向量检索和BM25检索
            async with measure_time("vector_search"):
                vector_task = self._vector_search(query.text, config.max_results, document_ids)
            async with measure_time("bm25_search"):
                bm25_task = self._bm25_search(query.text, config.max_results, document_ids)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_task, bm25_task, return_exceptions=True
            )
            
            # 处理异常
            if isinstance(vector_results, Exception):
                logger.error(f"向量检索失败: {vector_results}")
                vector_results = []
            
            if isinstance(bm25_results, Exception):
                logger.error(f"BM25检索失败: {bm25_results}")
                bm25_results = []
            
            # 融合结果
            fused_results = self._fuse_results(
                vector_results, bm25_results, config
            )
            
            # 重排序（如果启用）
            if config.enable_rerank:
                fused_results = await self._rerank_results(
                    query.text, fused_results
                )
            
            # 过滤和限制结果
            final_results = self._filter_and_limit_results(
                fused_results, config
            )
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            # 构建查询结果
            query_result = QueryResult(
                query_id=query.id,
                chunks=final_results,
                total_results=len(final_results),
                search_time=search_time,
                retrieval_method="hybrid"
            )
            
            logger.info(
                f"混合检索完成: 查询='{query.text}', "
                f"向量结果={len(vector_results)}, BM25结果={len(bm25_results)}, "
                f"最终结果={len(final_results)}, 耗时={search_time:.3f}s"
            )
            
            return query_result
            
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            raise
    
    async def _vector_search(
        self,
        query_text: str,
        limit: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """执行向量检索"""
        try:
            # 生成查询向量
            query_embedding = await self.embedding_service.embed_text(query_text)
            
            # 向量搜索
            results = await self.vector_service.search_similar(
                query_vector=query_embedding,
                limit=limit,
                document_ids=document_ids
            )
            
            # 标记检索方法
            for result in results:
                result["retrieval_method"] = "vector"
                result["vector_score"] = result["score"]
            
            return results
            
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return []
    
    async def _bm25_search(
        self,
        query_text: str,
        limit: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """执行BM25检索"""
        try:
            results = await self.bm25_service.search(
                query=query_text,
                limit=limit,
                document_ids=document_ids
            )
            
            # 标记检索方法
            for result in results:
                result["retrieval_method"] = "bm25"
                result["bm25_score"] = result["score"]
            
            return results
            
        except Exception as e:
            logger.error(f"BM25检索失败: {str(e)}")
            return []
    
    def _fuse_results(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        config: SearchConfig
    ) -> List[Dict]:
        """使用RRF算法融合检索结果"""
        # 创建结果字典，以文档ID为键
        result_dict = {}
        
        # 处理向量检索结果
        for i, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (60 + i + 1)  # RRF公式，k=60
            
            if doc_id not in result_dict:
                result_dict[doc_id] = result.copy()
                result_dict[doc_id]["vector_rank"] = i + 1
                result_dict[doc_id]["bm25_rank"] = None
                result_dict[doc_id]["vector_rrf"] = rrf_score
                result_dict[doc_id]["bm25_rrf"] = 0.0
            else:
                result_dict[doc_id]["vector_rank"] = i + 1
                result_dict[doc_id]["vector_rrf"] = rrf_score
        
        # 处理BM25检索结果
        for i, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (60 + i + 1)
            
            if doc_id not in result_dict:
                result_dict[doc_id] = result.copy()
                result_dict[doc_id]["vector_rank"] = None
                result_dict[doc_id]["bm25_rank"] = i + 1
                result_dict[doc_id]["vector_rrf"] = 0.0
                result_dict[doc_id]["bm25_rrf"] = rrf_score
            else:
                result_dict[doc_id]["bm25_rank"] = i + 1
                result_dict[doc_id]["bm25_rrf"] = rrf_score
        
        # 计算融合分数
        fused_results = []
        for doc_id, result in result_dict.items():
            # RRF融合分数
            rrf_score = result["vector_rrf"] + result["bm25_rrf"]
            
            # 加权融合分数（如果有原始分数）
            weighted_score = 0.0
            if "vector_score" in result:
                weighted_score += config.vector_weight * result["vector_score"]
            if "bm25_score" in result:
                # 归一化BM25分数
                normalized_bm25 = min(1.0, result["bm25_score"] / 10.0)
                weighted_score += config.bm25_weight * normalized_bm25
            
            # 最终分数结合RRF和加权分数
            final_score = 0.6 * rrf_score + 0.4 * weighted_score
            
            result["fusion_score"] = final_score
            result["rrf_score"] = rrf_score
            result["weighted_score"] = weighted_score
            
            fused_results.append(result)
        
        # 按融合分数排序
        fused_results.sort(key=lambda x: x["fusion_score"], reverse=True)
        
        return fused_results
    
    async def _rerank_results(
        self,
        query_text: str,
        results: List[Dict]
    ) -> List[Dict]:
        """重排序结果"""
        try:
            if not results:
                return results
            
            # 生成查询嵌入
            query_embedding = await self.embedding_service.embed_text(query_text)
            
            # 为每个结果计算语义相似度
            reranked_results = []
            for result in results:
                content = result.get("content", "")
                if content:
                    # 生成内容嵌入
                    content_embedding = await self.embedding_service.embed_text(content)
                    
                    # 计算相似度
                    similarity = await self.embedding_service.calculate_similarity(
                        query_embedding, content_embedding
                    )
                    
                    # 结合原始分数和语义相似度
                    original_score = result.get("fusion_score", 0.0)
                    rerank_score = 0.7 * original_score + 0.3 * similarity
                    
                    result["rerank_score"] = rerank_score
                    result["semantic_similarity"] = similarity
                    
                reranked_results.append(result)
            
            # 按重排序分数排序
            reranked_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return results
    
    def _filter_and_limit_results(
        self,
        results: List[Dict],
        config: SearchConfig
    ) -> List[Dict]:
        """过滤和限制结果"""
        # 过滤低分结果
        filtered_results = [
            result for result in results
            if result.get("fusion_score", 0) >= config.min_score_threshold
        ]
        
        # 限制结果数量
        return filtered_results[:config.max_results]
    
    async def search_vector_only(
        self,
        query: Query,
        limit: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> QueryResult:
        """仅使用向量检索"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            results = await self._vector_search(query.text, limit, document_ids)
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            return QueryResult(
                query_id=query.id,
                chunks=results,
                total_results=len(results),
                search_time=search_time,
                retrieval_method="vector"
            )
            
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            raise
    
    async def search_bm25_only(
        self,
        query: Query,
        limit: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> QueryResult:
        """仅使用BM25检索"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            results = await self._bm25_search(query.text, limit, document_ids)
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            return QueryResult(
                query_id=query.id,
                chunks=results,
                total_results=len(results),
                search_time=search_time,
                retrieval_method="bm25"
            )
            
        except Exception as e:
            logger.error(f"BM25检索失败: {str(e)}")
            raise
    
    async def search_bm25_only(
        self,
        query: Query,
        limit: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> QueryResult:
        """仅使用BM25检索"""
        try:
            start_time = asyncio.get_event_loop().time()

            results = await self._bm25_search(query.text, limit, document_ids)

            search_time = asyncio.get_event_loop().time() - start_time

            return QueryResult(
                query_id=query.id,
                chunks=results,
                total_results=len(results),
                search_time=search_time,
                retrieval_method="bm25"
            )

        except Exception as e:
            logger.error(f"BM25检索失败: {str(e)}")
            raise

    async def get_search_stats(self) -> Dict:
        """获取搜索统计信息"""
        try:
            vector_info = await self.vector_service.get_collection_info()
            bm25_stats = await self.bm25_service.get_index_stats()

            return {
                "vector_service": vector_info,
                "bm25_service": bm25_stats,
                "hybrid_search_enabled": True
            }

        except Exception as e:
            logger.error(f"获取搜索统计失败: {str(e)}")
            return {}