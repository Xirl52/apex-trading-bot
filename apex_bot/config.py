"""
Configuration module for APEX Trading Bot
Manages all configuration settings and environment variables
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for the trading bot"""
    
    # Neural Network Configuration
    NETWORK_TYPE = os.getenv("NETWORK_TYPE", "hybrid")  # hybrid, transformer, lstm
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
    HIDDEN_LAYERS = [512, 256, 128, 64]
    DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", "0.2"))
    
    # Self-Learning Configuration
    ENABLE_SELF_LEARNING = os.getenv("ENABLE_SELF_LEARNING", "true").lower() == "true"
    LEARNING_MODE = os.getenv("LEARNING_MODE", "continuous")  # continuous, episodic
    EXPLORATION_RATE = float(os.getenv("EXPLORATION_RATE", "0.3"))
    MEMORY_SIZE = int(os.getenv("MEMORY_SIZE", "100000"))
    
    # Data Access Configuration
    UNRESTRICTED_MODE = os.getenv("UNRESTRICTED_MODE", "true").lower() == "true"
    DATA_SOURCES = os.getenv("DATA_SOURCES", "alpha_vantage,yahoo,ccxt,alpaca").split(",")
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
    
    # Trading Configuration
    ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
    RISK_TOLERANCE = float(os.getenv("RISK_TOLERANCE", "0.02"))
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    
    # API Keys (stored in .env file)
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
    
    # Advanced Features
    ENABLE_SENTIMENT_ANALYSIS = os.getenv("ENABLE_SENTIMENT_ANALYSIS", "true").lower() == "true"
    ENABLE_NEWS_SCRAPING = os.getenv("ENABLE_NEWS_SCRAPING", "true").lower() == "true"
    ENABLE_SOCIAL_MEDIA = os.getenv("ENABLE_SOCIAL_MEDIA", "true").lower() == "true"
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Return all configuration as a dictionary"""
        return {
            k: v for k, v in cls.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings"""
        if cls.LEARNING_RATE <= 0 or cls.LEARNING_RATE > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        if cls.RISK_TOLERANCE < 0 or cls.RISK_TOLERANCE > 1:
            raise ValueError("Risk tolerance must be between 0 and 1")
        return True
