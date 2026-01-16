"""
Utility functions for the APEX Trading Bot
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a series of returns
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default 0.0)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve
    
    Args:
        equity_curve: Array of equity values over time
        
    Returns:
        Maximum drawdown as a percentage
    """
    if len(equity_curve) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return abs(np.min(drawdown))


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to [0, 1] range
    
    Args:
        features: Feature array
        
    Returns:
        Normalized features
    """
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    
    normalized = (features - min_val) / range_val
    return normalized


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV dataframe
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicators
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
    
    return df


def validate_data_quality(df: pd.DataFrame) -> bool:
    """
    Validate data quality
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data quality is acceptable
    """
    # Check for missing values
    if df.isnull().sum().sum() > len(df) * 0.1:
        logger.warning("More than 10% missing values detected")
        return False
    
    # Check for duplicate timestamps
    if df.index.duplicated().sum() > 0:
        logger.warning("Duplicate timestamps detected")
        return False
    
    # Check for negative prices or volumes
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns and (df[col] < 0).any():
            logger.warning(f"Negative values in {col}")
            return False
    
    return True


def format_currency(value: float, symbol: str = "$") -> str:
    """Format value as currency"""
    return f"{symbol}{value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value*100:.2f}%"
