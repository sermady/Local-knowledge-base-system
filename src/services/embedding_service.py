"""嵌入向量服务"""

import logging
import asyncio
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from sentence_transformers import SentenceTransformer
from src.config.settings import get_settings
from src.models.text_chunk import TextChunk
from src.utils.performance import monitor_performance, BatchProcessor

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """嵌入向量生成服务"""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._model_lock = asyncio.Lock()

        # 批处理器用于优化批量嵌入
        self.batch_processor = BatchProcessor(batch_size=32, max_wait_time=0.1)
    
    async def initialize(self):
        """初始化嵌入模型"""
        if self.model is None:
            async with self._model_lock:
                if self.model is None:
                    try:
                        logger.info(f"正在加载嵌入模型: {self.model_name}")
                        
                        # 在线程池中加载模型以避免阻塞
                        loop = asyncio.get_event_loop()
                        self.model = await loop.run_in_executor(
                            self.executor,
                            self._load_model_sync
                        )
                        
                        logger.info("嵌入模型加载完成")
                        
                    except Exception as e:
                        logger.error(f"嵌入模型加载失败: {str(e)}")
                        raise
    
    def _load_model_sync(self) -> SentenceTransformer:
        """同步加载模型"""
        return SentenceTransformer(self.model_name)
    
    @monitor_performance("embed_single_text")
    async def embed_text(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        await self.initialize()
        
        try:
            if not text or not text.strip():
                logger.warning("输入文本为空")
                return [0.0] * self.dimension
            
            # 在线程池中执行嵌入生成
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self._embed_text_sync,
                text.strip()
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"文本嵌入生成失败: {str(e)}")
            # 返回零向量作为fallback
            return [0.0] * self.dimension
    
    def _embed_text_sync(self, text: str) -> np.ndarray:
        """同步生成文本嵌入"""
        return self.model.encode(text, normalize_embeddings=True)
    
    @monitor_performance("embed_batch_texts")
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本嵌入向量"""
        await self.initialize()
        
        try:
            if not texts:
                return []
            
            # 过滤空文本
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                logger.warning("所有输入文本都为空")
                return [[0.0] * self.dimension] * len(texts)
            
            # 在线程池中执行批量嵌入生成
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                self._embed_texts_sync,
                valid_texts
            )
            
            # 转换为列表格式
            result = []
            valid_idx = 0
            for text in texts:
                if text and text.strip():
                    result.append(embeddings[valid_idx].tolist())
                    valid_idx += 1
                else:
                    result.append([0.0] * self.dimension)
            
            logger.info(f"批量生成 {len(texts)} 个文本的嵌入向量")
            return result
            
        except Exception as e:
            logger.error(f"批量文本嵌入生成失败: {str(e)}")
            # 返回零向量作为fallback
            return [[0.0] * self.dimension] * len(texts)
    
    def _embed_texts_sync(self, texts: List[str]) -> np.ndarray:
        """同步批量生成文本嵌入"""
        return self.model.encode(texts, normalize_embeddings=True)
    
    async def embed_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """为文本块生成嵌入向量"""
        try:
            # 提取文本内容
            texts = [chunk.content for chunk in chunks]
            
            # 批量生成嵌入向量
            embeddings = await self.embed_texts(texts)
            
            # 将嵌入向量赋值给文本块
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info(f"为 {len(chunks)} 个文本块生成嵌入向量")
            return chunks
            
        except Exception as e:
            logger.error(f"文本块嵌入生成失败: {str(e)}")
            raise
    
    async def get_model_info(self) -> dict:
        """获取模型信息"""
        await self.initialize()
        
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "is_loaded": self.model is not None
        }
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        try:
            embedding1 = await self.embed_text(text1)
            embedding2 = await self.embed_text(text2)
            
            # 计算余弦相似度
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            return 0.0
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
