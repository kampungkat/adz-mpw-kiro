"""
Configuration management for the AI Media Planner application.
Handles API keys, application settings, and environment configuration.
"""

import os
import streamlit as st
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration settings."""
    openai_api_key: str
    max_plans_generated: int = 3
    default_currency: str = "USD"
    cache_timeout_hours: int = 24
    max_file_size_mb: int = 10
    supported_file_formats: list = None
    
    def __post_init__(self):
        if self.supported_file_formats is None:
            self.supported_file_formats = ['.xlsx', '.xls']


class ConfigManager:
    """Manages application configuration and settings."""
    
    def __init__(self):
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from environment and Streamlit secrets."""
        if self._config is not None:
            return self._config
        
        # Try to get OpenAI API key from Streamlit secrets first, then environment
        openai_api_key = self._get_secret_or_env("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in "
                "Streamlit secrets or environment variables."
            )
        
        self._config = AppConfig(
            openai_api_key=openai_api_key,
            max_plans_generated=self._get_int_setting("MAX_PLANS_GENERATED", 3),
            default_currency=self._get_setting("DEFAULT_CURRENCY", "USD"),
            cache_timeout_hours=self._get_int_setting("CACHE_TIMEOUT_HOURS", 24),
            max_file_size_mb=self._get_int_setting("MAX_FILE_SIZE_MB", 10)
        )
        
        return self._config
    
    def _get_secret_or_env(self, key: str) -> Optional[str]:
        """Get value from Streamlit secrets or environment variables."""
        # Try Streamlit secrets first
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        
        # Fall back to environment variables
        return os.getenv(key)
    
    def _get_setting(self, key: str, default: str) -> str:
        """Get string setting with default value."""
        value = self._get_secret_or_env(key)
        return value if value is not None else default
    
    def _get_int_setting(self, key: str, default: int) -> int:
        """Get integer setting with default value."""
        value = self._get_secret_or_env(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
        return default
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key."""
        config = self.load_config()
        return config.openai_api_key
    
    def get_max_plans(self) -> int:
        """Get maximum number of plans to generate."""
        config = self.load_config()
        return config.max_plans_generated
    
    def get_cache_timeout(self) -> int:
        """Get cache timeout in hours."""
        config = self.load_config()
        return config.cache_timeout_hours
    
    def is_valid_file_format(self, filename: str) -> bool:
        """Check if file format is supported."""
        config = self.load_config()
        return any(filename.lower().endswith(fmt) for fmt in config.supported_file_formats)
    
    def get_max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        config = self.load_config()
        return config.max_file_size_mb * 1024 * 1024


# Global configuration manager instance
config_manager = ConfigManager()