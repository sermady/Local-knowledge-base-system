"""API中间件"""

import logging
import time
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def request_validation_middleware(request: Request, call_next: Callable) -> Response:
    """请求验证中间件"""
    start_time = time.time()
    
    # 记录请求信息
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent", "")
        }
    )
    
    try:
        # 检查请求大小限制
        content_length = request.headers.get("content-length")
        if content_length:
            content_length = int(content_length)
            max_size = 100 * 1024 * 1024  # 100MB
            if content_length > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"请求体过大，最大允许 {max_size // (1024*1024)}MB"
                )
        
        # 处理请求
        response = await call_next(request)
        
        # 记录响应信息
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} - {process_time:.3f}s",
            extra={
                "status_code": response.status_code,
                "process_time": process_time,
                "path": request.url.path
            }
        )
        
        # 添加响应头
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = "1.0"
        
        return response
        
    except HTTPException as e:
        # 处理HTTP异常
        process_time = time.time() - start_time
        logger.warning(
            f"HTTP Exception: {e.status_code} - {e.detail}",
            extra={
                "status_code": e.status_code,
                "detail": e.detail,
                "process_time": process_time,
                "path": request.url.path
            }
        )
        
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": e.detail,
                "status_code": e.status_code,
                "path": request.url.path,
                "timestamp": time.time()
            },
            headers={
                "X-Process-Time": str(process_time),
                "X-API-Version": "1.0"
            }
        )
        
    except Exception as e:
        # 处理未预期的异常
        process_time = time.time() - start_time
        logger.error(
            f"Unexpected error: {str(e)}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time": process_time,
                "path": request.url.path
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "内部服务器错误",
                "status_code": 500,
                "path": request.url.path,
                "timestamp": time.time()
            },
            headers={
                "X-Process-Time": str(process_time),
                "X-API-Version": "1.0"
            }
        )


async def rate_limiting_middleware(request: Request, call_next: Callable) -> Response:
    """简单的速率限制中间件"""
    # 这里可以实现更复杂的速率限制逻辑
    # 目前只是一个占位符
    
    client_ip = request.client.host if request.client else "unknown"
    
    # 简单的内存存储（生产环境应该使用Redis等）
    if not hasattr(rate_limiting_middleware, "requests"):
        rate_limiting_middleware.requests = {}
    
    current_time = time.time()
    window_size = 60  # 1分钟窗口
    max_requests = 100  # 每分钟最多100个请求
    
    # 清理过期记录
    rate_limiting_middleware.requests = {
        ip: timestamps for ip, timestamps in rate_limiting_middleware.requests.items()
        if any(t > current_time - window_size for t in timestamps)
    }
    
    # 检查当前IP的请求次数
    if client_ip not in rate_limiting_middleware.requests:
        rate_limiting_middleware.requests[client_ip] = []
    
    # 过滤窗口内的请求
    recent_requests = [
        t for t in rate_limiting_middleware.requests[client_ip]
        if t > current_time - window_size
    ]
    
    if len(recent_requests) >= max_requests:
        logger.warning(
            f"Rate limit exceeded for IP: {client_ip}",
            extra={"client_ip": client_ip, "request_count": len(recent_requests)}
        )
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "请求过于频繁，请稍后再试",
                "status_code": 429,
                "retry_after": window_size
            }
        )
    
    # 记录当前请求
    rate_limiting_middleware.requests[client_ip] = recent_requests + [current_time]
    
    return await call_next(request)


async def cors_middleware(request: Request, call_next: Callable) -> Response:
    """CORS中间件"""
    response = await call_next(request)
    
    # 添加CORS头
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    
    return response


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """安全头中间件"""
    response = await call_next(request)
    
    # 添加安全头
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response
