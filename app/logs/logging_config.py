#!/usr/bin/env python3
"""
Logging configuration for the TikTok compliance system.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# Global logging configuration
_LOGGING_SETUP = False
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FILE = None
_ROOT_LOGGER_CONFIGURED = False

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, uses default timestamped filename.
    
    Returns:
        Configured logger instance
    """
    global _LOGGING_SETUP, _DEFAULT_LOG_LEVEL, _DEFAULT_LOG_FILE, _ROOT_LOGGER_CONFIGURED
    
    # Store global settings
    _DEFAULT_LOG_LEVEL = log_level
    _DEFAULT_LOG_FILE = log_file
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent
    logs_dir.mkdir(exist_ok=True)
    
    # Set default log file if none provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"tiktok_compliance_{timestamp}.log"
    
    # Only configure root logger once to avoid conflicts
    if not _ROOT_LOGGER_CONFIGURED:
        # Create root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        _ROOT_LOGGER_CONFIGURED = True
        
        # Log the setup
        root_logger.info(f"Root logging configured with level: {log_level}")
        root_logger.info(f"Log file: {log_file}")
    
    # Mark logging as set up
    _LOGGING_SETUP = True
    
    # Create and return the main application logger
    main_logger = logging.getLogger("tiktok_compliance")
    main_logger.info(f"Logging system initialized with level: {log_level}")
    main_logger.info(f"Log file: {log_file}")
    
    return main_logger

def get_logger(name: str = "tiktok_compliance") -> logging.Logger:
    """
    Get a logger instance with the specified name.
    Automatically sets up file logging if not already configured.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance with file handlers
    """
    # Ensure logging is set up
    if not _LOGGING_SETUP:
        setup_logging(_DEFAULT_LOG_LEVEL, _DEFAULT_LOG_FILE)
    
    logger = logging.getLogger(name)
    
    # Check if this logger already has handlers
    if not logger.handlers:
        # Get the root logger's handlers and add them to this logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            # Create a copy of the handler to avoid conflicts
            if isinstance(handler, logging.FileHandler):
                # For file handlers, create a new one with the same file
                try:
                    new_handler = logging.FileHandler(handler.baseFilename, mode='a', encoding='utf-8')
                    new_handler.setLevel(handler.level)
                    new_handler.setFormatter(handler.formatter)
                    logger.addHandler(new_handler)
                except Exception as e:
                    # If file handler creation fails, fall back to console only
                    console_handler = logging.StreamHandler()
                    console_handler.setLevel(logging.INFO)
                    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    logger.addHandler(console_handler)
                    logger.warning(f"Failed to create file handler: {e}, using console only")
            elif isinstance(handler, logging.StreamHandler):
                # For console handlers, create a new one
                new_handler = logging.StreamHandler()
                new_handler.setLevel(handler.level)
                new_handler.setFormatter(handler.formatter)
                logger.addHandler(new_handler)
    
    # Ensure logger level is set appropriately
    if logger.level == logging.NOTSET:
        logger.setLevel(getattr(logging, _DEFAULT_LOG_LEVEL.upper()))
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger

def ensure_logging_setup():
    """
    Ensure logging is set up even if setup_logging() wasn't called.
    This is called automatically when needed.
    """
    if not _LOGGING_SETUP:
        setup_logging(_DEFAULT_LOG_LEVEL, _DEFAULT_LOG_FILE)

def get_current_log_file() -> str:
    """
    Get the current log file path being used.
    
    Returns:
        Path to current log file
    """
    if not _LOGGING_SETUP:
        ensure_logging_setup()
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    
    return "No log file configured"

def rotate_log_file():
    """
    Create a new log file with current timestamp.
    Useful for long-running applications.
    
    Returns:
        Path to new log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_log_file = Path(__file__).parent / f"tiktok_compliance_{timestamp}.log"
    
    # Set up new logging with the new file
    setup_logging(_DEFAULT_LOG_LEVEL, str(new_log_file))
    
    return str(new_log_file)

def force_log_file_creation():
    """
    Force creation of a new log file and ensure all loggers have file handlers.
    Useful for debugging logging issues.
    """
    global _ROOT_LOGGER_CONFIGURED
    _ROOT_LOGGER_CONFIGURED = False  # Allow reconfiguration
    
    # Create new log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_log_file = Path(__file__).parent / f"tiktok_compliance_{timestamp}.log"
    
    # Set up logging with new file
    setup_logging(_DEFAULT_LOG_LEVEL, str(new_log_file))
    
    # Force all existing loggers to get new handlers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            # Clear existing handlers and let get_logger recreate them
            logger.handlers.clear()
    
    return str(new_log_file)

# Auto-setup logging when module is imported
ensure_logging_setup()
