"""缓存服务实现"""

import logging
import asyncio
import sqlite3
import json
import hashlib
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timezone

from src.config.settings import get_settings
from src.models.cache import CacheEntry, QueryCache, CacheStats, CacheConfig
from src.models.search import QAResponse

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheService:
    """SQLite缓存服务"""
    
    def __init__(self):
        self.db_path = Path(settings.cache_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = CacheConfig(
            default_ttl=settings.cache_ttl,
            max_entries=settings.cache_max_size
        )
        
        self._db_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """初始化缓存数据库"""
        if not self._initialized:
            async with self._db_lock:
                if not self._initialized:
                    try:
                        await self._create_tables()
                        await self._cleanup_expired_cache()
                        self._initialized = True
                        logger.info("缓存服务初始化完成")
                        
                    except Exception as e:
                        logger.error(f"缓存服务初始化失败: {str(e)}")
                        raise
    
    async def _create_tables(self):
        """创建缓存表"""
        def create_tables_sync():
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # 创建查询缓存表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS query_cache (
                        query_hash TEXT PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        response_data TEXT NOT NULL,
                        document_ids TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        ttl INTEGER NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP NOT NULL
                    )
                ''')
                
                # 创建缓存统计表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_stats (
                        id INTEGER PRIMARY KEY,
                        total_entries INTEGER DEFAULT 0,
                        hit_count INTEGER DEFAULT 0,
                        miss_count INTEGER DEFAULT 0,
                        eviction_count INTEGER DEFAULT 0,
                        expired_count INTEGER DEFAULT 0,
                        last_updated TIMESTAMP NOT NULL
                    )
                ''')
                
                # 初始化统计记录
                cursor.execute('''
                    INSERT OR IGNORE INTO cache_stats (id, last_updated)
                    VALUES (1, ?)
                ''', (datetime.now(timezone.utc),))
                
                conn.commit()
                
            finally:
                conn.close()
        
        # 在线程池中执行数据库操作
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, create_tables_sync)
    
    def _generate_query_hash(self, query_text: str, document_ids: List[str] = None) -> str:
        """生成查询哈希值"""
        content = query_text
        if document_ids:
            content += "|" + "|".join(sorted(document_ids))
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def get_cached_result(self, query_text: str, document_ids: List[str] = None) -> Optional[QAResponse]:
        """获取缓存的查询结果"""
        await self.initialize()
        
        try:
            query_hash = self._generate_query_hash(query_text, document_ids)
            
            def get_cache_sync() -> Optional[Dict]:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # 查询缓存
                    cursor.execute('''
                        SELECT response_data, created_at, ttl, hit_count
                        FROM query_cache
                        WHERE query_hash = ?
                    ''', (query_hash,))
                    
                    result = cursor.fetchone()
                    if not result:
                        return None
                    
                    response_data, created_at, ttl, hit_count = result
                    
                    # 检查是否过期
                    created_time = datetime.fromisoformat(created_at)
                    if (datetime.now(timezone.utc) - created_time).total_seconds() > ttl:
                        # 删除过期缓存
                        cursor.execute('DELETE FROM query_cache WHERE query_hash = ?', (query_hash,))
                        conn.commit()
                        return None
                    
                    # 更新访问统计
                    cursor.execute('''
                        UPDATE query_cache
                        SET hit_count = hit_count + 1, last_accessed = ?
                        WHERE query_hash = ?
                    ''', (datetime.now(timezone.utc), query_hash))
                    
                    conn.commit()
                    
                    return {
                        'response_data': response_data,
                        'hit_count': hit_count + 1
                    }
                    
                finally:
                    conn.close()
            
            # 在线程池中执行数据库操作
            loop = asyncio.get_event_loop()
            cache_data = await loop.run_in_executor(None, get_cache_sync)
            
            if cache_data:
                # 更新统计
                await self._update_stats('hit')
                
                # 反序列化响应数据
                response_dict = json.loads(cache_data['response_data'])
                response = QAResponse(**response_dict)
                response.cached = True
                
                logger.info(f"缓存命中: {query_hash[:8]}...")
                return response
            else:
                # 更新统计
                await self._update_stats('miss')
                return None
                
        except Exception as e:
            logger.error(f"获取缓存失败: {str(e)}")
            return None
    
    async def cache_result(
        self,
        query_text: str,
        result: QAResponse,
        document_ids: List[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """缓存查询结果"""
        await self.initialize()
        
        try:
            query_hash = self._generate_query_hash(query_text, document_ids)
            ttl = ttl or self.config.default_ttl
            
            def cache_result_sync() -> bool:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # 序列化响应数据
                    response_dict = result.model_dump()
                    response_data = json.dumps(response_dict, ensure_ascii=False)
                    
                    # 插入或更新缓存
                    cursor.execute('''
                        INSERT OR REPLACE INTO query_cache
                        (query_hash, query_text, response_data, document_ids, created_at, ttl, hit_count, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    ''', (
                        query_hash,
                        query_text,
                        response_data,
                        json.dumps(document_ids or []),
                        datetime.now(timezone.utc),
                        ttl,
                        datetime.now(timezone.utc)
                    ))
                    
                    conn.commit()
                    return True
                    
                finally:
                    conn.close()
            
            # 在线程池中执行数据库操作
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, cache_result_sync)
            
            if success:
                logger.info(f"结果已缓存: {query_hash[:8]}...")
                
                # 检查是否需要清理缓存
                await self._check_and_cleanup()
            
            return success
            
        except Exception as e:
            logger.error(f"缓存结果失败: {str(e)}")
            return False
    
    async def invalidate_cache(self, document_id: str) -> bool:
        """使相关文档的缓存失效"""
        await self.initialize()
        
        try:
            def invalidate_sync() -> int:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # 查找包含指定文档ID的缓存
                    cursor.execute('SELECT query_hash, document_ids FROM query_cache')
                    rows = cursor.fetchall()
                    
                    to_delete = []
                    for query_hash, document_ids_json in rows:
                        try:
                            document_ids = json.loads(document_ids_json)
                            if document_id in document_ids:
                                to_delete.append(query_hash)
                        except json.JSONDecodeError:
                            continue
                    
                    # 删除相关缓存
                    if to_delete:
                        placeholders = ','.join(['?'] * len(to_delete))
                        cursor.execute(f'DELETE FROM query_cache WHERE query_hash IN ({placeholders})', to_delete)
                        conn.commit()
                    
                    return len(to_delete)
                    
                finally:
                    conn.close()
            
            # 在线程池中执行数据库操作
            loop = asyncio.get_event_loop()
            deleted_count = await loop.run_in_executor(None, invalidate_sync)
            
            logger.info(f"文档 {document_id} 相关的 {deleted_count} 个缓存已失效")
            return True
            
        except Exception as e:
            logger.error(f"缓存失效操作失败: {str(e)}")
            return False
    
    async def get_cache_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        await self.initialize()
        
        try:
            def get_stats_sync() -> Dict:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # 获取统计信息
                    cursor.execute('SELECT * FROM cache_stats WHERE id = 1')
                    stats_row = cursor.fetchone()
                    
                    # 获取当前缓存条目数
                    cursor.execute('SELECT COUNT(*) FROM query_cache')
                    current_entries = cursor.fetchone()[0]
                    
                    if stats_row:
                        return {
                            'total_entries': current_entries,
                            'hit_count': stats_row[2],
                            'miss_count': stats_row[3],
                            'eviction_count': stats_row[4],
                            'expired_count': stats_row[5]
                        }
                    else:
                        return {
                            'total_entries': current_entries,
                            'hit_count': 0,
                            'miss_count': 0,
                            'eviction_count': 0,
                            'expired_count': 0
                        }
                        
                finally:
                    conn.close()
            
            # 在线程池中执行数据库操作
            loop = asyncio.get_event_loop()
            stats_data = await loop.run_in_executor(None, get_stats_sync)
            
            return CacheStats(**stats_data)
            
        except Exception as e:
            logger.error(f"获取缓存统计失败: {str(e)}")
            return CacheStats()
    
    async def cleanup_expired_cache(self) -> int:
        """清理过期缓存"""
        await self.initialize()
        
        try:
            def cleanup_sync() -> int:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # 删除过期缓存
                    current_time = datetime.now(timezone.utc)
                    cursor.execute('''
                        DELETE FROM query_cache
                        WHERE datetime(created_at, '+' || ttl || ' seconds') < ?
                    ''', (current_time,))
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    return deleted_count
                    
                finally:
                    conn.close()
            
            # 在线程池中执行数据库操作
            loop = asyncio.get_event_loop()
            deleted_count = await loop.run_in_executor(None, cleanup_sync)
            
            if deleted_count > 0:
                await self._update_stats('expired', deleted_count)
                logger.info(f"清理了 {deleted_count} 个过期缓存")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理过期缓存失败: {str(e)}")
            return 0
    
    async def _cleanup_expired_cache(self):
        """内部清理过期缓存"""
        await self.cleanup_expired_cache()
    
    async def _check_and_cleanup(self):
        """检查并清理缓存"""
        stats = await self.get_cache_stats()
        
        if stats.total_entries > self.config.max_entries:
            # 删除最旧的缓存条目
            await self._evict_old_entries(stats.total_entries - self.config.max_entries)
    
    async def _evict_old_entries(self, count: int):
        """驱逐旧的缓存条目"""
        try:
            def evict_sync() -> int:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    # 删除最旧的条目
                    cursor.execute('''
                        DELETE FROM query_cache
                        WHERE query_hash IN (
                            SELECT query_hash FROM query_cache
                            ORDER BY last_accessed ASC
                            LIMIT ?
                        )
                    ''', (count,))
                    
                    evicted_count = cursor.rowcount
                    conn.commit()
                    
                    return evicted_count
                    
                finally:
                    conn.close()
            
            # 在线程池中执行数据库操作
            loop = asyncio.get_event_loop()
            evicted_count = await loop.run_in_executor(None, evict_sync)
            
            if evicted_count > 0:
                await self._update_stats('eviction', evicted_count)
                logger.info(f"驱逐了 {evicted_count} 个旧缓存条目")
            
        except Exception as e:
            logger.error(f"驱逐缓存条目失败: {str(e)}")
    
    async def _update_stats(self, operation: str, count: int = 1):
        """更新统计信息"""
        try:
            def update_stats_sync():
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    
                    if operation == 'hit':
                        cursor.execute('''
                            UPDATE cache_stats
                            SET hit_count = hit_count + ?, last_updated = ?
                            WHERE id = 1
                        ''', (count, datetime.now(timezone.utc)))
                    elif operation == 'miss':
                        cursor.execute('''
                            UPDATE cache_stats
                            SET miss_count = miss_count + ?, last_updated = ?
                            WHERE id = 1
                        ''', (count, datetime.now(timezone.utc)))
                    elif operation == 'eviction':
                        cursor.execute('''
                            UPDATE cache_stats
                            SET eviction_count = eviction_count + ?, last_updated = ?
                            WHERE id = 1
                        ''', (count, datetime.now(timezone.utc)))
                    elif operation == 'expired':
                        cursor.execute('''
                            UPDATE cache_stats
                            SET expired_count = expired_count + ?, last_updated = ?
                            WHERE id = 1
                        ''', (count, datetime.now(timezone.utc)))
                    
                    conn.commit()
                    
                finally:
                    conn.close()
            
            # 在线程池中执行数据库操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, update_stats_sync)
            
        except Exception as e:
            logger.error(f"更新统计信息失败: {str(e)}")
