"""Kimi2问答服务"""

import logging
import asyncio
import time
from typing import List, Optional, Dict, Any
import json

import openai
from src.config.settings import get_settings
from src.models.text_chunk import TextChunk
from src.models.search import QAResponse, Citation
from src.models.conversation import Message, Conversation

logger = logging.getLogger(__name__)
settings = get_settings()


class QAService:
    """基于Kimi2 API的问答服务"""
    
    def __init__(self):
        self.client = None
        self.model_name = "moonshot-v1-8k"
        self._client_lock = asyncio.Lock()
        
        # 约束性提示词模板
        self.system_prompt = """你是一个专业的知识库助手。请严格遵循以下规则：

1. 仅基于提供的文档内容回答问题，不得使用任何外部知识
2. 如果文档中没有相关信息，明确回答"根据提供的文档内容，我无法找到相关信息"
3. 回答时请引用具体的文档片段，并标注来源
4. 保持回答的准确性和客观性
5. 如果文档内容不完整或模糊，请说明这一点

请基于以下文档内容回答用户问题：

{context}

用户问题：{question}"""
    
    async def initialize(self):
        """初始化Kimi2客户端"""
        if self.client is None:
            async with self._client_lock:
                if self.client is None:
                    try:
                        if not settings.moonshot_api_key:
                            raise ValueError("MOONSHOT_API_KEY 未设置")
                        
                        self.client = openai.AsyncOpenAI(
                            api_key=settings.moonshot_api_key,
                            base_url=settings.moonshot_base_url
                        )
                        
                        logger.info("Kimi2客户端初始化完成")
                        
                    except Exception as e:
                        logger.error(f"Kimi2客户端初始化失败: {str(e)}")
                        raise
    
    async def generate_answer(
        self,
        question: str,
        context_chunks: List[TextChunk],
        conversation_id: Optional[str] = None
    ) -> QAResponse:
        """生成问答响应"""
        start_time = time.time()
        
        try:
            await self.initialize()
            
            # 检查是否有上下文
            if not context_chunks:
                return await self.handle_no_context(question, conversation_id)
            
            # 构建上下文
            context = self._build_context(context_chunks)
            
            # 构建提示词
            prompt = self.system_prompt.format(
                context=context,
                question=question
            )
            
            # 调用Kimi2 API
            response = await self._call_kimi_api(prompt)
            
            # 提取来源引用
            sources = self._extract_sources(response, context_chunks)
            
            # 验证答案来源
            has_context = self._validate_answer_source(response, context_chunks)
            
            processing_time = time.time() - start_time
            
            return QAResponse(
                question=question,
                answer=response,
                sources=sources,
                confidence=self._calculate_confidence(response, context_chunks),
                processing_time=processing_time,
                cached=False,
                conversation_id=conversation_id,
                has_context=has_context,
                api_usage={
                    "model": self.model_name,
                    "tokens_used": len(prompt.split()) + len(response.split())  # 粗略估算
                }
            )
            
        except Exception as e:
            logger.error(f"问答生成失败: {str(e)}")
            processing_time = time.time() - start_time
            
            return QAResponse(
                question=question,
                answer=f"抱歉，处理您的问题时出现错误：{str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=processing_time,
                cached=False,
                conversation_id=conversation_id,
                has_context=False
            )
    
    def _build_context(self, chunks: List[TextChunk]) -> str:
        """构建上下文字符串"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_part = f"文档片段 {i}:\n"
            context_part += f"来源: {chunk.metadata.document_title or '未知文档'}\n"
            if chunk.metadata.page_number:
                context_part += f"页码: {chunk.metadata.page_number}\n"
            if chunk.metadata.section_title:
                context_part += f"章节: {chunk.metadata.section_title}\n"
            context_part += f"内容: {chunk.content}\n"
            context_part += "-" * 50 + "\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def _call_kimi_api(self, prompt: str) -> str:
        """调用Kimi2 API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 降低随机性，提高一致性
                max_tokens=2000,
                timeout=settings.request_timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Kimi2 API调用失败: {str(e)}")
            raise
    
    def _extract_sources(self, answer: str, chunks: List[TextChunk]) -> List[Citation]:
        """提取答案中的来源引用"""
        sources = []
        
        for i, chunk in enumerate(chunks):
            # 简单的匹配策略：检查答案中是否包含文档片段的关键内容
            chunk_words = set(chunk.content.split())
            answer_words = set(answer.split())
            
            # 计算重叠度
            overlap = len(chunk_words.intersection(answer_words))
            if overlap > 3:  # 如果有足够的重叠词汇
                citation = Citation(
                    chunk_id=str(chunk.id),
                    document_id=chunk.document_id,
                    document_title=chunk.metadata.document_title or "未知文档",
                    page_number=chunk.metadata.page_number,
                    section_title=chunk.metadata.section_title,
                    quoted_text=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    relevance_score=min(overlap / len(chunk_words), 1.0) if chunk_words else 0.0
                )
                sources.append(citation)
        
        # 按相关性排序
        sources.sort(key=lambda x: x.relevance_score, reverse=True)
        return sources[:5]  # 最多返回5个来源
    
    def _validate_answer_source(self, answer: str, chunks: List[TextChunk]) -> bool:
        """验证答案是否基于提供的文档内容"""
        # 检查是否包含"无法找到相关信息"等表示没有答案的关键词
        no_answer_keywords = [
            "无法找到相关信息",
            "没有相关信息",
            "文档中没有",
            "无法回答",
            "不清楚",
            "没有提及"
        ]
        
        for keyword in no_answer_keywords:
            if keyword in answer:
                return False
        
        # 简单验证：检查答案是否与文档内容有足够的重叠
        all_chunk_content = " ".join([chunk.content for chunk in chunks])
        chunk_words = set(all_chunk_content.split())
        answer_words = set(answer.split())
        
        overlap_ratio = len(chunk_words.intersection(answer_words)) / len(answer_words) if answer_words else 0
        return overlap_ratio > 0.1  # 至少10%的重叠
    
    def _calculate_confidence(self, answer: str, chunks: List[TextChunk]) -> float:
        """计算答案的置信度"""
        if not self._validate_answer_source(answer, chunks):
            return 0.0
        
        # 基于多个因素计算置信度
        factors = []
        
        # 1. 答案长度因子（适中的长度通常更可靠）
        answer_length = len(answer.split())
        if 20 <= answer_length <= 200:
            factors.append(0.8)
        elif 10 <= answer_length <= 300:
            factors.append(0.6)
        else:
            factors.append(0.4)
        
        # 2. 上下文匹配因子
        all_content = " ".join([chunk.content for chunk in chunks])
        content_words = set(all_content.split())
        answer_words = set(answer.split())
        
        if content_words and answer_words:
            overlap_ratio = len(content_words.intersection(answer_words)) / len(answer_words)
            factors.append(min(overlap_ratio * 2, 1.0))
        else:
            factors.append(0.0)
        
        # 3. 来源数量因子
        sources_count = len(self._extract_sources(answer, chunks))
        if sources_count >= 2:
            factors.append(0.9)
        elif sources_count == 1:
            factors.append(0.7)
        else:
            factors.append(0.3)
        
        # 计算加权平均
        return sum(factors) / len(factors) if factors else 0.0
    
    async def handle_no_context(self, question: str, conversation_id: Optional[str] = None) -> QAResponse:
        """处理没有上下文的情况"""
        return QAResponse(
            question=question,
            answer="抱歉，我在您的文档库中没有找到与此问题相关的信息。请确保已上传相关文档，或尝试使用不同的关键词重新提问。",
            sources=[],
            confidence=0.0,
            processing_time=0.0,
            cached=False,
            conversation_id=conversation_id,
            has_context=False
        )
    
    async def multi_turn_conversation(
        self,
        conversation: Conversation,
        new_question: str,
        context_chunks: List[TextChunk]
    ) -> QAResponse:
        """多轮对话处理"""
        # 构建对话历史
        conversation_history = []
        for message in conversation.messages[-5:]:  # 只保留最近5轮对话
            if message.role == "user":
                conversation_history.append(f"用户: {message.content}")
            else:
                conversation_history.append(f"助手: {message.content}")
        
        # 在问题前添加对话历史
        enhanced_question = f"对话历史:\n{chr(10).join(conversation_history)}\n\n当前问题: {new_question}"
        
        return await self.generate_answer(
            enhanced_question,
            context_chunks,
            str(conversation.id)
        )
