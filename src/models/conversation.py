"""
对话相关数据模型
Conversation-related data models
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import BaseDataModel
from .search import QAResponse


class Message(BaseDataModel):
    """消息数据模型"""
    
    conversation_id: str = Field(..., description="对话ID")
    role: str = Field(..., description="角色（user/assistant）")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="时间戳")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """验证角色"""
        valid_roles = {'user', 'assistant', 'system'}
        if v not in valid_roles:
            raise ValueError(f"无效的角色: {v}")
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """验证内容"""
        if not v or len(v.strip()) == 0:
            raise ValueError("消息内容不能为空")
        return v.strip()
    
    def is_user_message(self) -> bool:
        """是否为用户消息"""
        return self.role == 'user'
    
    def is_assistant_message(self) -> bool:
        """是否为助手消息"""
        return self.role == 'assistant'


class Conversation(BaseDataModel):
    """对话数据模型"""
    
    user_id: Optional[str] = Field(None, description="用户ID")
    title: Optional[str] = Field(None, description="对话标题")
    messages: List[Message] = Field(default_factory=list, description="消息列表")
    context_documents: List[str] = Field(
        default_factory=list, 
        description="上下文文档ID列表"
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        description="最后活动时间"
    )
    is_active: bool = Field(default=True, description="是否活跃")
    
    def add_message(self, message: Message) -> None:
        """添加消息"""
        if message.conversation_id != self.id:
            message.conversation_id = self.id
        self.messages.append(message)
        self.last_activity = datetime.now(timezone.utc)
        self.update_timestamp()
    
    def get_message_count(self) -> int:
        """获取消息数量"""
        return len(self.messages)
    
    def get_last_message(self) -> Optional[Message]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None
    
    def get_user_messages(self) -> List[Message]:
        """获取用户消息"""
        return [msg for msg in self.messages if msg.is_user_message()]
    
    def get_assistant_messages(self) -> List[Message]:
        """获取助手消息"""
        return [msg for msg in self.messages if msg.is_assistant_message()]
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """获取最近的消息"""
        return self.messages[-count:] if count > 0 else []
    
    def generate_title(self) -> str:
        """生成对话标题"""
        if not self.messages:
            return "新对话"
        
        first_user_msg = next(
            (msg for msg in self.messages if msg.is_user_message()), 
            None
        )
        if first_user_msg:
            # 取前30个字符作为标题
            title = first_user_msg.content[:30]
            if len(first_user_msg.content) > 30:
                title += "..."
            return title
        
        return f"对话 {self.id[:8]}"
    
    def update_title(self) -> None:
        """更新对话标题"""
        if not self.title:
            self.title = self.generate_title()
            self.update_timestamp()
    
    def add_context_document(self, document_id: str) -> None:
        """添加上下文文档"""
        if document_id not in self.context_documents:
            self.context_documents.append(document_id)
            self.update_timestamp()
    
    def remove_context_document(self, document_id: str) -> None:
        """移除上下文文档"""
        if document_id in self.context_documents:
            self.context_documents.remove(document_id)
            self.update_timestamp()
    
    def deactivate(self) -> None:
        """停用对话"""
        self.is_active = False
        self.update_timestamp()


class ConversationSummary(BaseDataModel):
    """对话摘要数据模型"""
    
    conversation_id: str = Field(..., description="对话ID")
    summary: str = Field(..., description="摘要内容")
    key_topics: List[str] = Field(default_factory=list, description="关键话题")
    mentioned_documents: List[str] = Field(
        default_factory=list, 
        description="提及的文档ID列表"
    )
    message_count: int = Field(default=0, ge=0, description="消息数量")
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    
    @field_validator('summary')
    @classmethod
    def validate_summary(cls, v: str) -> str:
        """验证摘要"""
        if not v or len(v.strip()) == 0:
            raise ValueError("摘要不能为空")
        return v.strip()
    
    @field_validator('key_topics')
    @classmethod
    def validate_key_topics(cls, v: List[str]) -> List[str]:
        """验证关键话题"""
        return list(set(topic.strip() for topic in v if topic.strip()))
    
    def add_topic(self, topic: str) -> None:
        """添加关键话题"""
        topic = topic.strip()
        if topic and topic not in self.key_topics:
            self.key_topics.append(topic)
            self.update_timestamp()
    
    def get_duration(self) -> float:
        """获取对话持续时间（秒）"""
        return (self.end_time - self.start_time).total_seconds()


class ConversationContext(BaseDataModel):
    """对话上下文数据模型"""
    
    conversation_id: str = Field(..., description="对话ID")
    current_topic: Optional[str] = Field(None, description="当前话题")
    context_window: List[Message] = Field(
        default_factory=list, 
        description="上下文窗口消息"
    )
    relevant_documents: List[str] = Field(
        default_factory=list, 
        description="相关文档ID列表"
    )
    user_intent: Optional[str] = Field(None, description="用户意图")
    last_qa_response: Optional[QAResponse] = Field(None, description="最后的问答响应")
    
    def update_context_window(self, messages: List[Message], max_size: int = 10) -> None:
        """更新上下文窗口"""
        self.context_window = messages[-max_size:] if max_size > 0 else messages
        self.update_timestamp()
    
    def add_relevant_document(self, document_id: str) -> None:
        """添加相关文档"""
        if document_id not in self.relevant_documents:
            self.relevant_documents.append(document_id)
            self.update_timestamp()
    
    def update_topic(self, topic: str) -> None:
        """更新当前话题"""
        self.current_topic = topic.strip() if topic else None
        self.update_timestamp()
    
    def update_intent(self, intent: str) -> None:
        """更新用户意图"""
        self.user_intent = intent.strip() if intent else None
        self.update_timestamp()
    
    def get_context_length(self) -> int:
        """获取上下文长度"""
        return len(self.context_window)
    
    def has_recent_qa(self) -> bool:
        """是否有最近的问答"""
        return self.last_qa_response is not None