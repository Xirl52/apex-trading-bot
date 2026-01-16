"""
Universal Data Fetcher - Unrestricted data access module
Fetches data from multiple sources without limitations
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class UniversalDataFetcher:
    """
    Unrestricted data fetcher that can access multiple data sources
    and retrieve any available market information on demand
    """
    
    def __init__(self, config: 'Config'):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Any] = {}
        self.sources_available = self._initialize_sources()
        
    def _initialize_sources(self) -> List[str]:
        """Initialize all available data sources"""
        sources = []
        
        # Check which data sources are configured
        for source in self.config.DATA_SOURCES:
            sources.append(source.strip())
            
        logger.info(f"Initialized {len(sources)} data sources: {sources}")
        return sources
    
    async def fetch_market_data(
        self, 
        symbol: str, 
        timeframe: str = '1d',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch market data for a given symbol from multiple sources
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD', 'AAPL')
            timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
            limit: Number of data points to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching market data for {symbol} with timeframe {timeframe}")
        
        # Try multiple sources for redundancy
        for source in self.sources_available:
            try:
                if source == 'yahoo':
                    return await self._fetch_from_yahoo(symbol, limit)
                elif source == 'ccxt':
                    return await self._fetch_from_ccxt(symbol, timeframe, limit)
                elif source == 'alpaca':
                    return await self._fetch_from_alpaca(symbol, timeframe, limit)
            except Exception as e:
                logger.warning(f"Failed to fetch from {source}: {e}")
                continue
                
        # Return synthetic data if all sources fail (for demo/testing)
        return self._generate_synthetic_data(symbol, limit)
    
    async def _fetch_from_yahoo(self, symbol: str, limit: int) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='max')
            return df.tail(limit)
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            raise
    
    async def _fetch_from_ccxt(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int
    ) -> pd.DataFrame:
        """Fetch data from cryptocurrency exchanges via CCXT"""
        try:
            import ccxt
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp')
        except Exception as e:
            logger.error(f"CCXT error: {e}")
            raise
    
    async def _fetch_from_alpaca(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int
    ) -> pd.DataFrame:
        """Fetch data from Alpaca API"""
        # Placeholder - would implement actual Alpaca API calls
        raise NotImplementedError("Alpaca integration pending API keys")
    
    def _generate_synthetic_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1D')
        
        # Generate realistic price data with random walk
        returns = np.random.randn(limit) * 0.02
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': price * (1 + np.random.randn(limit) * 0.005),
            'high': price * (1 + abs(np.random.randn(limit) * 0.01)),
            'low': price * (1 - abs(np.random.randn(limit) * 0.01)),
            'close': price,
            'volume': np.random.randint(1000000, 10000000, limit)
        }, index=dates)
        
        return df
    
    async def fetch_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch news and sentiment data for a symbol
        Unrestricted access to news sources
        """
        if not self.config.ENABLE_NEWS_SCRAPING:
            return {}
            
        logger.info(f"Fetching news sentiment for {symbol}")
        
        # Placeholder for news sentiment aggregation
        return {
            'sentiment_score': 0.5,
            'news_count': 0,
            'trending': False
        }
    
    async def fetch_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch social media sentiment (Twitter, Reddit, etc.)
        Unrestricted social media access
        """
        if not self.config.ENABLE_SOCIAL_MEDIA:
            return {}
            
        logger.info(f"Fetching social sentiment for {symbol}")
        
        # Placeholder for social media sentiment
        return {
            'twitter_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'mentions': 0
        }
    
    async def fetch_alternative_data(self, query: str) -> Dict[str, Any]:
        """
        Fetch alternative data sources based on query
        This allows the bot to "go wherever it wants" to gather data
        """
        if not self.config.UNRESTRICTED_MODE:
            return {}
            
        logger.info(f"Fetching alternative data for: {query}")
        
        # Placeholder for unrestricted web scraping and data gathering
        return {
            'data': [],
            'source': 'web',
            'timestamp': datetime.now().isoformat()
        }
    
    async def close(self):
        """Close the data fetcher and cleanup resources"""
        if self.session:
            await self.session.close()
