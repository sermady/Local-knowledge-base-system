"""向量数据库服务"""

import logging
from typing import List, Dict, Optional, Tuple
import asyncio
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from src.models.text_chunk import TextChunk
from src.models.query import QueryResult
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorService:
    """向量数据库服务"""
    
    def __init__(self):
        self.client = None
        self.collection_name = "knowledge_base"
        self._client_lock = asyncio.Lock()
    
    async def initialize(self):
        """初始化Qdrant客户端"""
        if self.client is None:
            async with self._client_lock:
                if self.client is None:
                    try:
                        self.client = QdrantClient(
                            host=settings.qdrant_host,
                            port=settings.qdrant_port,
                            timeout=30
                        )
                        
                        # 创建集合
                        await self._create_collection()
                        logger.info("Qdrant客户端初始化完成")
                        
                    except Exception as e:
                        logger.error(f"Qdrant初始化失败: {str(e)}")
                        raise
    
    async def _create_collection(self):
        """创建向量集合"""
        try:
            # 检查集合是否存在
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建向量集合: {self.collection_name}")
            else:
                logger.info(f"向量集合已存在: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            raise
    
    async def store_chunks(self, chunks: List[TextChunk]) -> bool:
        """存储文本块向量"""
        await self.initialize()
        
        try:
            points = []
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"文本块 {chunk.id} 没有嵌入向量，跳过")
                    continue
                
                point = PointStruct(
                    id=str(chunk.id),
                    vector=chunk.embedding,
                    payload={
                        "document_id": str(chunk.document_id),
                        "content": chunk.content,
                        "content_length": chunk.content_length,
                        "chunk_index": chunk.metadata.chunk_index,
                        "total_chunks": chunk.metadata.total_chunks,
                        "page_number": chunk.metadata.page_number,
                        "section_title": chunk.metadata.section_title,
                        "language": chunk.metadata.language,
                        "confidence_score": chunk.metadata.confidence_score,
                        "created_at": chunk.created_at.isoformat()
                    }
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"成功存储 {len(points)} 个向量点")
                return True
            else:
                logger.warning("没有有效的向量点可存储")
                return False
                
        except Exception as e:
            logger.error(f"存储向量失败: {str(e)}")
            raise
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """搜索相似向量"""
        await self.initialize()
        
        try:
            # 构建过滤条件
            query_filter = None
            if document_ids:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=doc_id)
                        ) for doc_id in document_ids
                    ]
                )
            
            # 执行搜索
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # 转换结果格式
            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                results.append(result)
            
            logger.info(f"向量搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            raise
    
    async def delete_by_document(self, document_id: str) -> bool:
        """删除指定文档的所有向量"""
        await self.initialize()
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"删除文档 {document_id} 的所有向量")
            return True
            
        except Exception as e:
            logger.error(f"删除向量失败: {str(e)}")
            return False
    
    async def update_chunk(self, chunk: TextChunk) -> bool:
        """更新单个文本块"""
        await self.initialize()
        
        try:
            if chunk.embedding is None:
                logger.warning(f"文本块 {chunk.id} 没有嵌入向量")
                return False
            
            point = PointStruct(
                id=str(chunk.id),
                vector=chunk.embedding,
                payload={
                    "document_id": str(chunk.document_id),
                    "content": chunk.content,
                    "content_length": chunk.content_length,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks,
                    "page_number": chunk.metadata.page_number,
                    "section_title": chunk.metadata.section_title,
                    "language": chunk.metadata.language,
                    "confidence_score": chunk.metadata.confidence_score,
                    "created_at": chunk.created_at.isoformat()
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"更新文本块向量: {chunk.id}")
            return True
            
        except Exception as e:
            logger.error(f"更新向量失败: {str(e)}")
            return False
    
    async def get_collection_info(self) -> Dict:
        """获取集合信息"""
        await self.initialize()
        
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            return {}
    
    async def scroll_points(self, limit: int = 100, offset: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
        """分页获取向量点"""
        await self.initialize()
        
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = []
            for point in result[0]:
                points.append({
                    "id": point.id,
                    "payload": point.payload
                })
            
            next_offset = result[1]
            return points, next_offset
            
        except Exception as e:
            logger.error(f"分页获取向量点失败: {str(e)}")
            return [], None