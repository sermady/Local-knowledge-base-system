"""
数据模型单元测试
Unit tests for data models
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from src.models import (
    BaseDataModel, ProcessingStatus, EntityType,
    Document, DocumentInfo, ParsedContent,
    TextChunk, ChunkMetadata, Vector,
    Entity, Mention, Relation, KnowledgeGraph,
    SearchResult, SearchQuery, SearchResponse, Citation, QAResponse,
    Message, Conversation, ConversationSummary, ConversationContext,
    CacheEntry, QueryCache, CacheStats, CacheConfig, CacheOperation
)


class TestBaseDataModel:
    """基础数据模型测试"""
    
    def test_base_model_creation(self):
        """测试基础模型创建"""
        model = BaseDataModel()
        assert model.id is not None
        assert len(model.id) > 0
        assert isinstance(model.created_at, datetime)
        assert isinstance(model.updated_at, datetime)
    
    def test_update_timestamp(self):
        """测试时间戳更新"""
        model = BaseDataModel()
        original_time = model.updated_at
        model.update_timestamp()
        assert model.updated_at > original_time
    
    def test_id_validation(self):
        """测试ID验证"""
        with pytest.raises(ValueError):
            BaseDataModel(id="")
        with pytest.raises(ValueError):
            BaseDataModel(id="   ")


class TestDocument:
    """文档模型测试"""
    
    def test_document_creation(self):
        """测试文档创建"""
        doc = Document(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            content_hash="a" * 64
        )
        assert doc.filename == "test.pdf"
        assert doc.file_type == "pdf"
        assert doc.file_size == 1024
        assert doc.processing_status == ProcessingStatus.PENDING
        assert doc.version == 1
    
    def test_filename_validation(self):
        """测试文件名验证"""
        with pytest.raises(ValueError):
            Document(filename="", file_type="pdf", file_size=1024, content_hash="a" * 64)
        
        with pytest.raises(ValueError):
            Document(filename="a" * 256, file_type="pdf", file_size=1024, content_hash="a" * 64)
    
    def test_file_type_validation(self):
        """测试文件类型验证"""
        with pytest.raises(ValueError):
            Document(filename="test.xyz", file_type="xyz", file_size=1024, content_hash="a" * 64)
    
    def test_content_hash_validation(self):
        """测试内容哈希验证"""
        with pytest.raises(ValueError):
            Document(filename="test.pdf", file_type="pdf", file_size=1024, content_hash="")
        
        with pytest.raises(ValueError):
            Document(filename="test.pdf", file_type="pdf", file_size=1024, content_hash="short")
    
    def test_tag_operations(self):
        """测试标签操作"""
        doc = Document(
            filename="test.pdf",
            file_type="pdf", 
            file_size=1024,
            content_hash="a" * 64
        )
        
        doc.add_tag("important")
        assert "important" in doc.tags
        
        doc.add_tag("important")  # 重复添加
        assert doc.tags.count("important") == 1
        
        doc.remove_tag("important")
        assert "important" not in doc.tags
    
    def test_version_increment(self):
        """测试版本递增"""
        doc = Document(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024, 
            content_hash="a" * 64
        )
        
        original_version = doc.version
        doc.increment_version()
        assert doc.version == original_version + 1


class TestTextChunk:
    """文本块模型测试"""
    
    def test_text_chunk_creation(self):
        """测试文本块创建"""
        chunk = TextChunk(
            document_id="doc1",
            content="这是一个测试文本块",
            chunk_index=0,
            start_position=0,
            end_position=10
        )
        assert chunk.document_id == "doc1"
        assert chunk.content == "这是一个测试文本块"
        assert chunk.chunk_index == 0
    
    def test_content_validation(self):
        """测试内容验证"""
        with pytest.raises(ValueError):
            TextChunk(
                document_id="doc1",
                content="",
                chunk_index=0,
                start_position=0,
                end_position=10
            )
        
        with pytest.raises(ValueError):
            TextChunk(
                document_id="doc1",
                content="a" * 10001,  # 超过长度限制
                chunk_index=0,
                start_position=0,
                end_position=10
            )
    
    def test_position_validation(self):
        """测试位置验证"""
        with pytest.raises(ValueError):
            TextChunk(
                document_id="doc1",
                content="test",
                chunk_index=0,
                start_position=10,
                end_position=5  # 结束位置小于起始位置
            )
    
    def test_embedding_operations(self):
        """测试向量操作"""
        chunk = TextChunk(
            document_id="doc1",
            content="test",
            chunk_index=0,
            start_position=0,
            end_position=4
        )
        
        assert not chunk.has_embedding()
        
        embedding = [0.1] * 768
        chunk.set_embedding(embedding)
        assert chunk.has_embedding()
        assert len(chunk.embedding) == 768
    
    def test_similarity_calculation(self):
        """测试相似度计算"""
        # 使用768维向量（支持的维度）
        embedding = [1.0] + [0.0] * 767
        chunk = TextChunk(
            document_id="doc1",
            content="test",
            chunk_index=0,
            start_position=0,
            end_position=4,
            embedding=embedding
        )
        
        other_embedding = [1.0] + [0.0] * 767
        similarity = chunk.calculate_similarity(other_embedding)
        assert abs(similarity - 1.0) < 1e-6  # 完全相同的向量


class TestEntity:
    """实体模型测试"""
    
    def test_entity_creation(self):
        """测试实体创建"""
        entity = Entity(
            name="张三",
            entity_type=EntityType.PERSON
        )
        assert entity.name == "张三"
        assert entity.entity_type == EntityType.PERSON
        assert len(entity.mentions) == 0
    
    def test_mention_operations(self):
        """测试提及操作"""
        entity = Entity(name="张三", entity_type=EntityType.PERSON)
        mention = Mention(
            text="张三",
            start_position=0,
            end_position=2,
            chunk_id="chunk1",
            document_id="doc1",
            confidence=0.9
        )
        
        entity.add_mention(mention)
        assert len(entity.mentions) == 1
        assert entity.get_mention_count() == 1
    
    def test_alias_operations(self):
        """测试别名操作"""
        entity = Entity(name="张三", entity_type=EntityType.PERSON)
        
        entity.add_alias("小张")
        assert "小张" in entity.aliases
        
        entity.add_alias("张三")  # 不应该添加与名称相同的别名
        assert "张三" not in entity.aliases


class TestSearchModels:
    """搜索模型测试"""
    
    def test_search_result_creation(self):
        """测试搜索结果创建"""
        result = SearchResult(
            chunk_id="chunk1",
            document_id="doc1",
            content="测试内容",
            score=0.85
        )
        assert result.chunk_id == "chunk1"
        assert result.score == 0.85
    
    def test_search_query_validation(self):
        """测试搜索查询验证"""
        with pytest.raises(ValueError):
            SearchQuery(query_text="")
        
        with pytest.raises(ValueError):
            SearchQuery(query_text="test", query_type="invalid")
    
    def test_qa_response_creation(self):
        """测试问答响应创建"""
        response = QAResponse(
            question="什么是AI？",
            answer="AI是人工智能的缩写"
        )
        assert response.question == "什么是AI？"
        assert response.answer == "AI是人工智能的缩写"
        assert len(response.sources) == 0
    
    def test_citation_formatting(self):
        """测试引用格式化"""
        citation = Citation(
            document_id="doc1",
            document_title="AI基础",
            chunk_id="chunk1",
            quoted_text="AI是人工智能",
            relevance_score=0.9,
            page_number=1
        )
        
        formatted = citation.get_citation_format()
        assert "《AI基础》" in formatted
        assert "第1页" in formatted


class TestConversationModels:
    """对话模型测试"""
    
    def test_message_creation(self):
        """测试消息创建"""
        message = Message(
            conversation_id="conv1",
            role="user",
            content="你好"
        )
        assert message.role == "user"
        assert message.content == "你好"
        assert message.is_user_message()
    
    def test_message_role_validation(self):
        """测试消息角色验证"""
        with pytest.raises(ValueError):
            Message(
                conversation_id="conv1",
                role="invalid",
                content="test"
            )
    
    def test_conversation_operations(self):
        """测试对话操作"""
        conv = Conversation()
        
        message = Message(
            conversation_id=conv.id,
            role="user",
            content="你好"
        )
        
        conv.add_message(message)
        assert conv.get_message_count() == 1
        assert conv.get_last_message() == message
    
    def test_conversation_title_generation(self):
        """测试对话标题生成"""
        conv = Conversation()
        
        message = Message(
            conversation_id=conv.id,
            role="user",
            content="这是一个很长的问题，用来测试标题生成功能是否正常工作"
        )
        
        conv.add_message(message)
        title = conv.generate_title()
        assert len(title) <= 33  # 30个字符 + "..."


class TestCacheModels:
    """缓存模型测试"""
    
    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            ttl=3600
        )
        assert entry.key == "test_key"
        assert entry.ttl == 3600
        assert not entry.is_expired()
    
    def test_cache_entry_expiration(self):
        """测试缓存条目过期"""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            ttl=1
        )
        
        # 模拟时间过去
        entry.created_at = datetime.now(timezone.utc) - timedelta(seconds=2)
        assert entry.is_expired()
        assert entry.get_remaining_ttl() == 0
    
    def test_query_cache_creation(self):
        """测试查询缓存创建"""
        response = QAResponse(
            question="test",
            answer="test answer"
        )
        
        cache = QueryCache(
            query_hash="a" * 64,
            query_text="test query",
            response=response
        )
        assert cache.query_hash == "a" * 64
        assert cache.hit_count == 0
    
    def test_cache_stats_operations(self):
        """测试缓存统计操作"""
        stats = CacheStats()
        
        stats.record_hit()
        stats.record_miss()
        
        assert stats.hit_count == 1
        assert stats.miss_count == 1
        assert abs(stats.get_hit_rate() - 0.5) < 1e-6
    
    def test_cache_config_validation(self):
        """测试缓存配置验证"""
        with pytest.raises(ValueError):
            CacheConfig(eviction_policy="invalid")
        
        config = CacheConfig(eviction_policy="lru")
        assert config.eviction_policy == "lru"


class TestModelSerialization:
    """模型序列化测试"""
    
    def test_document_serialization(self):
        """测试文档序列化"""
        doc = Document(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            content_hash="a" * 64
        )
        
        # 测试转换为字典
        doc_dict = doc.model_dump()
        assert doc_dict["filename"] == "test.pdf"
        assert doc_dict["file_type"] == "pdf"
        
        # 测试从字典创建
        new_doc = Document(**doc_dict)
        assert new_doc.filename == doc.filename
        assert new_doc.file_type == doc.file_type
    
    def test_json_serialization(self):
        """测试JSON序列化"""
        chunk = TextChunk(
            document_id="doc1",
            content="test",
            chunk_index=0,
            start_position=0,
            end_position=4
        )
        
        # 测试JSON序列化
        json_str = chunk.model_dump_json()
        assert isinstance(json_str, str)
        assert "doc1" in json_str
        
        # 测试从JSON创建
        new_chunk = TextChunk.model_validate_json(json_str)
        assert new_chunk.document_id == chunk.document_id
        assert new_chunk.content == chunk.content


class TestModelValidation:
    """模型验证测试"""
    
    def test_field_validation(self):
        """测试字段验证"""
        # 测试必填字段
        with pytest.raises(ValueError):
            Document()  # 缺少必填字段
        
        # 测试数值范围验证
        with pytest.raises(ValueError):
            Document(
                filename="test.pdf",
                file_type="pdf",
                file_size=-1,  # 负数
                content_hash="a" * 64
            )
    
    def test_custom_validators(self):
        """测试自定义验证器"""
        # 测试文件类型验证
        with pytest.raises(ValueError):
            Document(
                filename="test.unknown",
                file_type="unknown",
                file_size=1024,
                content_hash="a" * 64
            )
        
        # 测试位置验证
        with pytest.raises(ValueError):
            TextChunk(
                document_id="doc1",
                content="test",
                chunk_index=0,
                start_position=10,
                end_position=5
            )


if __name__ == "__main__":
    pytest.main([__file__])