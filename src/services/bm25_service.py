"""BM25关键词检索服务"""

import logging
import pickle
from typing import List, Dict, Optional, Tuple
import jieba
import jieba.analyse
from rank_bm25 import BM25Okapi
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.models.text_chunk import TextChunk
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BM25Service:
    """BM25关键词检索服务"""
    
    def __init__(self):
        self.bm25_index = None
        self.documents = []  # 存储文档内容和元数据
        self.index_file = Path("data/bm25_index.pkl")
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._index_lock = asyncio.Lock()
        
        # 初始化jieba
        jieba.initialize()
        # 设置自定义词典（如果有的话）
        self._setup_custom_dict()
    
    def _setup_custom_dict(self):
        """设置自定义词典"""
        custom_dict_path = Path("data/custom_dict.txt")
        if custom_dict_path.exists():
            jieba.load_userdict(str(custom_dict_path))
            logger.info("加载自定义词典")
    
    async def build_index(self, chunks: List[TextChunk]):
        """构建BM25索引"""
        async with self._index_lock:
            try:
                logger.info(f"开始构建BM25索引，文档数量: {len(chunks)}")
                
                # 在线程池中执行分词和索引构建
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self._build_index_sync,
                    chunks
                )
                
                # 保存索引
                await self._save_index()
                
                logger.info("BM25索引构建完成")
                
            except Exception as e:
                logger.error(f"构建BM25索引失败: {str(e)}")
                raise
    
    def _build_index_sync(self, chunks: List[TextChunk]):
        """同步构建索引"""
        # 准备文档数据
        self.documents = []
        tokenized_docs = []
        
        for chunk in chunks:
            # 分词
            tokens = self._tokenize_text(chunk.content)
            tokenized_docs.append(tokens)
            
            # 存储文档元数据
            doc_info = {
                "id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "content": chunk.content,
                "tokens": tokens,
                "metadata": {
                    "chunk_index": chunk.metadata.chunk_index,
                    "page_number": chunk.metadata.page_number,
                    "section_title": chunk.metadata.section_title,
                    "language": chunk.metadata.language
                }
            }
            self.documents.append(doc_info)
        
        # 构建BM25索引
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """文本分词"""
        # 使用jieba进行中文分词
        tokens = list(jieba.cut(text, cut_all=False))
        
        # 过滤停用词和短词
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            if (len(token) > 1 and 
                not token.isspace() and 
                not self._is_stopword(token)):
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def _is_stopword(self, word: str) -> bool:
        """判断是否为停用词"""
        # 简单的停用词列表
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他',
            '她', '它', '们', '这个', '那个', '什么', '怎么', '为什么'
        }
        return word in stopwords
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """执行BM25搜索"""
        if self.bm25_index is None:
            await self._load_index()
            if self.bm25_index is None:
                logger.warning("BM25索引未构建")
                return []
        
        try:
            # 查询分词
            query_tokens = self._tokenize_text(query)
            if not query_tokens:
                return []
            
            # 在线程池中执行搜索
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._search_sync,
                query_tokens,
                limit,
                document_ids
            )
            
            logger.info(f"BM25搜索完成，查询: '{query}', 返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"BM25搜索失败: {str(e)}")
            return []
    
    def _search_sync(
        self,
        query_tokens: List[str],
        limit: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """同步执行搜索"""
        # 获取BM25分数
        scores = self.bm25_index.get_scores(query_tokens)
        
        # 创建结果列表
        results = []
        for i, score in enumerate(scores):
            if i >= len(self.documents):
                continue
                
            doc = self.documents[i]
            
            # 如果指定了文档ID过滤
            if document_ids and doc["document_id"] not in document_ids:
                continue
            
            if score > 0:  # 只返回有相关性的结果
                result = {
                    "id": doc["id"],
                    "document_id": doc["document_id"],
                    "content": doc["content"],
                    "score": float(score),
                    "metadata": doc["metadata"],
                    "matched_terms": self._get_matched_terms(query_tokens, doc["tokens"])
                }
                results.append(result)
        
        # 按分数排序并限制结果数量
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _get_matched_terms(self, query_tokens: List[str], doc_tokens: List[str]) -> List[str]:
        """获取匹配的词项"""
        doc_token_set = set(doc_tokens)
        matched = []
        for token in query_tokens:
            if token in doc_token_set:
                matched.append(token)
        return matched
    
    async def add_documents(self, chunks: List[TextChunk]):
        """增量添加文档"""
        async with self._index_lock:
            try:
                # 处理新文档
                new_docs = []
                new_tokenized_docs = []
                
                for chunk in chunks:
                    tokens = self._tokenize_text(chunk.content)
                    new_tokenized_docs.append(tokens)
                    
                    doc_info = {
                        "id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "content": chunk.content,
                        "tokens": tokens,
                        "metadata": {
                            "chunk_index": chunk.metadata.chunk_index,
                            "page_number": chunk.metadata.page_number,
                            "section_title": chunk.metadata.section_title,
                            "language": chunk.metadata.language
                        }
                    }
                    new_docs.append(doc_info)
                
                # 更新文档列表
                self.documents.extend(new_docs)
                
                # 重建索引（BM25Okapi不支持增量更新）
                all_tokenized_docs = [doc["tokens"] for doc in self.documents]
                self.bm25_index = BM25Okapi(all_tokenized_docs)
                
                # 保存索引
                await self._save_index()
                
                logger.info(f"增量添加 {len(chunks)} 个文档到BM25索引")
                
            except Exception as e:
                logger.error(f"增量添加文档失败: {str(e)}")
                raise
    
    async def remove_documents(self, document_ids: List[str]):
        """删除文档"""
        async with self._index_lock:
            try:
                # 过滤掉要删除的文档
                original_count = len(self.documents)
                self.documents = [
                    doc for doc in self.documents 
                    if doc["document_id"] not in document_ids
                ]
                
                removed_count = original_count - len(self.documents)
                
                if removed_count > 0:
                    # 重建索引
                    if self.documents:
                        all_tokenized_docs = [doc["tokens"] for doc in self.documents]
                        self.bm25_index = BM25Okapi(all_tokenized_docs)
                    else:
                        self.bm25_index = None
                    
                    # 保存索引
                    await self._save_index()
                    
                    logger.info(f"从BM25索引中删除 {removed_count} 个文档")
                
            except Exception as e:
                logger.error(f"删除文档失败: {str(e)}")
                raise
    
    async def _save_index(self):
        """保存索引到文件"""
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            
            index_data = {
                "bm25_index": self.bm25_index,
                "documents": self.documents
            }
            
            with open(self.index_file, 'wb') as f:
                pickle.dump(index_data, f)
                
            logger.info("BM25索引已保存")
            
        except Exception as e:
            logger.error(f"保存BM25索引失败: {str(e)}")
    
    async def _load_index(self):
        """从文件加载索引"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'rb') as f:
                    index_data = pickle.load(f)
                
                self.bm25_index = index_data.get("bm25_index")
                self.documents = index_data.get("documents", [])
                
                logger.info("BM25索引已加载")
            else:
                logger.info("BM25索引文件不存在")
                
        except Exception as e:
            logger.error(f"加载BM25索引失败: {str(e)}")
    
    async def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        if self.bm25_index is None:
            await self._load_index()
        
        return {
            "document_count": len(self.documents) if self.documents else 0,
            "index_exists": self.bm25_index is not None,
            "index_file_exists": self.index_file.exists(),
            "total_tokens": sum(len(doc["tokens"]) for doc in self.documents) if self.documents else 0
        }