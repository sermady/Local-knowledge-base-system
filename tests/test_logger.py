"""
Tests for logging functionality.
"""
import pytest
import logging
from pathlib import Path
from src.utils.logger import setup_logging, get_logger, LoggerMixin, log_error


def test_setup_logging():
    """Test logging setup."""
    setup_logging()
    
    # Check if logs directory is created
    assert Path("logs").exists()
    
    # Test logger creation
    logger = get_logger("test")
    assert logger is not None


def test_logger_mixin():
    """Test LoggerMixin functionality."""
    
    class TestClass(LoggerMixin):
        def test_method(self):
            self.logger.info("Test message")
            return "success"
    
    test_obj = TestClass()
    result = test_obj.test_method()
    
    assert result == "success"
    assert hasattr(test_obj, 'logger')


def test_log_error():
    """Test error logging function."""
    setup_logging()
    
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_error(e, {"context": "test"})
        # Should not raise any exceptions


def test_get_logger():
    """Test logger retrieval."""
    setup_logging()
    
    logger1 = get_logger("test1")
    logger2 = get_logger("test2")
    logger3 = get_logger("test1")  # Same name
    
    assert logger1 is not None
    assert logger2 is not None
    # Note: structlog creates new instances but they should have same name
    assert logger1._context == logger3._context