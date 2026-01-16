"""
APEX Trading Bot - Advanced Self-Learning Neural Network Trading System

This bot implements an unrestricted, self-learning neural network that can:
- Access multiple data sources and trading platforms
- Learn from market patterns autonomously
- Make intelligent trading decisions
- Adapt to changing market conditions
"""

from .neural_network import UnrestrictedNeuralNetwork
from .trading_agent import SelfLearningAgent
from .data_fetcher import UniversalDataFetcher
from .config import Config

__version__ = "1.0.0"
__all__ = [
    "UnrestrictedNeuralNetwork",
    "SelfLearningAgent", 
    "UniversalDataFetcher",
    "Config"
]
