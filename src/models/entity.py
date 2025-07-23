"""
实体相关数据模型
Entity-related data models
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import BaseDataModel, EntityType


class Mention(BaseDataModel):
    """实体提及"""
    
    text: str = Field(..., description="提及文本")
    start_position: int = Field(..., ge=0, description="起始位置")
    end_position: int = Field(..., ge=0, description="结束位置")
    chunk_id: str = Field(..., description="所在文本块ID")
    document_id: str = Field(..., description="所在文档ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    context: Optional[str] = Field(None, description="上下文")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """验证提及文本"""
        if not v or len(v.strip()) == 0:
            raise ValueError("提及文本不能为空")
        return v.strip()
    
    @field_validator('end_position')
    @classmethod
    def validate_positions(cls, v: int, info) -> int:
        """验证位置信息"""
        if info.data:
            start_pos = info.data.get('start_position', 0)
            if v <= start_pos:
                raise ValueError("结束位置必须大于起始位置")
        return v
    
    def get_text_length(self) -> int:
        """获取文本长度"""
        return len(self.text)
    
    def get_position_range(self) -> tuple[int, int]:
        """获取位置范围"""
        return (self.start_position, self.end_position)


class Entity(BaseDataModel):
    """实体数据模型"""
    
    name: str = Field(..., description="实体名称")
    entity_type: EntityType = Field(..., description="实体类型")
    mentions: List[Mention] = Field(default_factory=list, description="实体提及列表")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    aliases: List[str] = Field(default_factory=list, description="别名列表")
    description: Optional[str] = Field(None, description="实体描述")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="整体置信度")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """验证实体名称"""
        if not v or len(v.strip()) == 0:
            raise ValueError("实体名称不能为空")
        return v.strip()
    
    @field_validator('aliases')
    @classmethod
    def validate_aliases(cls, v: List[str]) -> List[str]:
        """验证别名列表"""
        return list(set(alias.strip() for alias in v if alias.strip()))
    
    def add_mention(self, mention: Mention) -> None:
        """添加实体提及"""
        if mention not in self.mentions:
            self.mentions.append(mention)
            self.update_timestamp()
    
    def add_alias(self, alias: str) -> None:
        """添加别名"""
        alias = alias.strip()
        if alias and alias not in self.aliases and alias != self.name:
            self.aliases.append(alias)
            self.update_timestamp()
    
    def get_mention_count(self) -> int:
        """获取提及次数"""
        return len(self.mentions)
    
    def get_documents(self) -> List[str]:
        """获取相关文档ID列表"""
        return list(set(mention.document_id for mention in self.mentions))
    
    def get_chunks(self) -> List[str]:
        """获取相关文本块ID列表"""
        return list(set(mention.chunk_id for mention in self.mentions))
    
    def calculate_average_confidence(self) -> float:
        """计算平均置信度"""
        if not self.mentions:
            return self.confidence
        return sum(mention.confidence for mention in self.mentions) / len(self.mentions)


class Relation(BaseDataModel):
    """实体关系数据模型"""
    
    source_entity_id: str = Field(..., description="源实体ID")
    target_entity_id: str = Field(..., description="目标实体ID")
    relation_type: str = Field(..., description="关系类型")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    evidence: List[str] = Field(default_factory=list, description="证据文本列表")
    context_chunks: List[str] = Field(
        default_factory=list, 
        description="相关文本块ID列表"
    )
    attributes: Dict[str, Any] = Field(default_factory=dict, description="关系属性")
    
    @field_validator('relation_type')
    @classmethod
    def validate_relation_type(cls, v: str) -> str:
        """验证关系类型"""
        if not v or len(v.strip()) == 0:
            raise ValueError("关系类型不能为空")
        return v.strip().lower()
    
    @field_validator('source_entity_id')
    @classmethod
    def validate_source_entity_id(cls, v: str) -> str:
        """验证源实体ID"""
        if not v or len(v.strip()) == 0:
            raise ValueError("源实体ID不能为空")
        return v.strip()
    
    @field_validator('target_entity_id')
    @classmethod
    def validate_target_entity_id(cls, v: str, info) -> str:
        """验证目标实体ID"""
        if not v or len(v.strip()) == 0:
            raise ValueError("目标实体ID不能为空")
        if info.data:
            source_id = info.data.get('source_entity_id')
            if source_id and v.strip() == source_id:
                raise ValueError("源实体和目标实体不能相同")
        return v.strip()
    
    def add_evidence(self, evidence: str) -> None:
        """添加证据"""
        evidence = evidence.strip()
        if evidence and evidence not in self.evidence:
            self.evidence.append(evidence)
            self.update_timestamp()
    
    def add_context_chunk(self, chunk_id: str) -> None:
        """添加相关文本块"""
        chunk_id = chunk_id.strip()
        if chunk_id and chunk_id not in self.context_chunks:
            self.context_chunks.append(chunk_id)
            self.update_timestamp()
    
    def get_evidence_count(self) -> int:
        """获取证据数量"""
        return len(self.evidence)
    
    def is_bidirectional(self) -> bool:
        """判断是否为双向关系"""
        bidirectional_relations = {
            'similar_to', 'related_to', 'connected_to', 
            'associated_with', 'colleague_of'
        }
        return self.relation_type in bidirectional_relations


class KnowledgeGraph(BaseDataModel):
    """知识图谱数据模型"""
    
    entities: Dict[str, Entity] = Field(default_factory=dict, description="实体字典")
    relations: List[Relation] = Field(default_factory=list, description="关系列表")
    document_ids: List[str] = Field(default_factory=list, description="相关文档ID列表")
    
    def add_entity(self, entity: Entity) -> None:
        """添加实体"""
        self.entities[entity.id] = entity
        self.update_timestamp()
    
    def add_relation(self, relation: Relation) -> None:
        """添加关系"""
        # 检查实体是否存在
        if (relation.source_entity_id not in self.entities or 
            relation.target_entity_id not in self.entities):
            raise ValueError("关系中的实体不存在于知识图谱中")
        
        self.relations.append(relation)
        self.update_timestamp()
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    def get_entity_relations(self, entity_id: str) -> List[Relation]:
        """获取实体的所有关系"""
        return [
            rel for rel in self.relations 
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]
    
    def get_related_entities(self, entity_id: str) -> List[str]:
        """获取相关实体ID列表"""
        related = set()
        for relation in self.get_entity_relations(entity_id):
            if relation.source_entity_id == entity_id:
                related.add(relation.target_entity_id)
            else:
                related.add(relation.source_entity_id)
        return list(related)
    
    def get_entity_count(self) -> int:
        """获取实体数量"""
        return len(self.entities)
    
    def get_relation_count(self) -> int:
        """获取关系数量"""
        return len(self.relations)
    
    def remove_entity(self, entity_id: str) -> None:
        """移除实体及其相关关系"""
        if entity_id in self.entities:
            del self.entities[entity_id]
            # 移除相关关系
            self.relations = [
                rel for rel in self.relations 
                if rel.source_entity_id != entity_id and rel.target_entity_id != entity_id
            ]
            self.update_timestamp()