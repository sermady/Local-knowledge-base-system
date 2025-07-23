"""性能监控工具"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    
    def add_measurement(self, duration: float, success: bool = True):
        """添加测量结果"""
        self.total_calls += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)
        
        if not success:
            self.error_count += 1
    
    def get_average_time(self) -> float:
        """获取平均时间"""
        return self.total_time / self.total_calls if self.total_calls > 0 else 0.0
    
    def get_recent_average(self) -> float:
        """获取最近的平均时间"""
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        return self.error_count / self.total_calls if self.total_calls > 0 else 0.0


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self._lock = threading.Lock()
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """记录操作性能"""
        with self._lock:
            self.metrics[operation_name].add_measurement(duration, success)
    
    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            if operation_name:
                if operation_name in self.metrics:
                    metrics = self.metrics[operation_name]
                    return {
                        "operation": operation_name,
                        "total_calls": metrics.total_calls,
                        "total_time": metrics.total_time,
                        "average_time": metrics.get_average_time(),
                        "recent_average": metrics.get_recent_average(),
                        "min_time": metrics.min_time if metrics.min_time != float('inf') else 0.0,
                        "max_time": metrics.max_time,
                        "error_count": metrics.error_count,
                        "error_rate": metrics.get_error_rate()
                    }
                else:
                    return {}
            else:
                result = {}
                for name, metrics in self.metrics.items():
                    result[name] = {
                        "total_calls": metrics.total_calls,
                        "average_time": metrics.get_average_time(),
                        "recent_average": metrics.get_recent_average(),
                        "error_rate": metrics.get_error_rate()
                    }
                return result
    
    def reset_metrics(self, operation_name: Optional[str] = None):
        """重置指标"""
        with self._lock:
            if operation_name:
                if operation_name in self.metrics:
                    del self.metrics[operation_name]
            else:
                self.metrics.clear()


# 全局性能监控器实例
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: Optional[str] = None):
    """性能监控装饰器"""
    def decorator(func: Callable):
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    performance_monitor.record_operation(name, duration, success)
                    
                    if duration > 1.0:  # 记录慢操作
                        logger.warning(f"Slow operation: {name} took {duration:.3f}s")
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    performance_monitor.record_operation(name, duration, success)
                    
                    if duration > 1.0:  # 记录慢操作
                        logger.warning(f"Slow operation: {name} took {duration:.3f}s")
            
            return sync_wrapper
    
    return decorator


@asynccontextmanager
async def measure_time(operation_name: str):
    """异步上下文管理器用于测量时间"""
    start_time = time.time()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.time() - start_time
        performance_monitor.record_operation(operation_name, duration, success)


class BatchProcessor:
    """批处理器，用于优化批量操作性能"""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.pending_futures = []
        self.last_batch_time = time.time()
        self._lock = asyncio.Lock()
        self._processing = False
    
    async def add_item(self, item: Any, processor: Callable) -> Any:
        """添加项目到批处理队列"""
        async with self._lock:
            future = asyncio.Future()
            self.pending_items.append((item, processor))
            self.pending_futures.append(future)
            
            # 检查是否需要处理批次
            should_process = (
                len(self.pending_items) >= self.batch_size or
                time.time() - self.last_batch_time >= self.max_wait_time
            )
            
            if should_process and not self._processing:
                asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """处理当前批次"""
        async with self._lock:
            if self._processing or not self.pending_items:
                return
            
            self._processing = True
            items = self.pending_items.copy()
            futures = self.pending_futures.copy()
            self.pending_items.clear()
            self.pending_futures.clear()
            self.last_batch_time = time.time()
        
        try:
            # 按处理器分组
            processor_groups = defaultdict(list)
            for i, (item, processor) in enumerate(items):
                processor_groups[processor].append((i, item))
            
            # 并行处理每个组
            results = [None] * len(items)
            tasks = []
            
            for processor, group_items in processor_groups.items():
                task = asyncio.create_task(
                    self._process_group(processor, group_items, results)
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 设置结果
            for i, future in enumerate(futures):
                if not future.done():
                    if i < len(results):
                        if isinstance(results[i], Exception):
                            future.set_exception(results[i])
                        else:
                            future.set_result(results[i])
                    else:
                        future.set_exception(RuntimeError("处理失败"))
        
        except Exception as e:
            # 设置所有未完成的future为异常
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        
        finally:
            async with self._lock:
                self._processing = False
    
    async def _process_group(self, processor: Callable, group_items: list, results: list):
        """处理一组项目"""
        try:
            items = [item for _, item in group_items]
            
            if asyncio.iscoroutinefunction(processor):
                group_results = await processor(items)
            else:
                group_results = processor(items)
            
            # 将结果放回正确的位置
            for (index, _), result in zip(group_items, group_results):
                results[index] = result
                
        except Exception as e:
            # 为这组的所有项目设置异常
            for index, _ in group_items:
                results[index] = e


class ConnectionPool:
    """简单的连接池实现"""
    
    def __init__(self, create_connection: Callable, max_size: int = 10):
        self.create_connection = create_connection
        self.max_size = max_size
        self.pool = asyncio.Queue(maxsize=max_size)
        self.current_size = 0
        self._lock = asyncio.Lock()
    
    async def get_connection(self):
        """获取连接"""
        try:
            # 尝试从池中获取连接
            connection = self.pool.get_nowait()
            return connection
        except asyncio.QueueEmpty:
            # 池为空，创建新连接
            async with self._lock:
                if self.current_size < self.max_size:
                    self.current_size += 1
                    if asyncio.iscoroutinefunction(self.create_connection):
                        connection = await self.create_connection()
                    else:
                        connection = self.create_connection()
                    return connection
                else:
                    # 等待连接可用
                    return await self.pool.get()
    
    async def return_connection(self, connection):
        """归还连接"""
        try:
            self.pool.put_nowait(connection)
        except asyncio.QueueFull:
            # 池已满，关闭连接
            if hasattr(connection, 'close'):
                if asyncio.iscoroutinefunction(connection.close):
                    await connection.close()
                else:
                    connection.close()
            
            async with self._lock:
                self.current_size -= 1
    
    @asynccontextmanager
    async def connection(self):
        """连接上下文管理器"""
        conn = await self.get_connection()
        try:
            yield conn
        finally:
            await self.return_connection(conn)
