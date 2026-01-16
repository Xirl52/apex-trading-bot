"""
Self-Learning Trading Agent
Autonomous agent that learns from experience and makes trading decisions
"""

import torch
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

from .neural_network import UnrestrictedNeuralNetwork, AdaptiveLearner
from .data_fetcher import UniversalDataFetcher
from .config import Config

logger = logging.getLogger(__name__)


class SelfLearningAgent:
    """
    Self-learning trading agent with unrestricted capabilities
    Uses reinforcement learning to continuously improve
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize neural network
        self.network = UnrestrictedNeuralNetwork(
            input_size=100,  # Will be adjusted based on features
            hidden_layers=config.HIDDEN_LAYERS,
            output_size=3,  # Buy, Sell, Hold
            dropout_rate=config.DROPOUT_RATE,
            network_type=config.NETWORK_TYPE
        ).to(self.device)
        
        # Adaptive learner
        self.learner = AdaptiveLearner(
            self.network,
            initial_lr=config.LEARNING_RATE
        )
        
        # Experience replay memory for self-learning
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.exploration_rate = config.EXPLORATION_RATE
        
        # Performance tracking
        self.total_rewards = 0.0
        self.episode_count = 0
        self.trade_history: List[Dict[str, Any]] = []
        
        # Data fetcher
        self.data_fetcher = UniversalDataFetcher(config)
        
        logger.info(f"Initialized self-learning agent on {self.device}")
    
    def decide_action(
        self, 
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """
        Decide trading action based on current state
        Uses epsilon-greedy for exploration vs exploitation
        
        Args:
            state: Current market state
            deterministic: If True, always pick best action
            
        Returns:
            action: 0=Hold, 1=Buy, 2=Sell
            confidence: Confidence in the action
        """
        # Exploration: random action
        if not deterministic and random.random() < self.exploration_rate:
            action = random.randint(0, 2)
            confidence = 0.33
            logger.debug(f"Exploring: random action {action}")
            return action, confidence
        
        # Exploitation: use neural network
        self.network.eval()
        with torch.no_grad():
            state = state.to(self.device)
            if len(state.shape) == 2:
                state = state.unsqueeze(0)  # Add batch dimension
                
            action_probs, state_value, _ = self.network(state)
            action = torch.argmax(action_probs, dim=-1).item()
            confidence = action_probs[0, action].item()
            
        logger.debug(f"Action: {action}, Confidence: {confidence:.3f}")
        return action, confidence
    
    def learn_from_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """
        Learn from a single experience (state, action, reward, next_state)
        This is the core of self-learning
        """
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Only learn if we have enough experiences
        if len(self.memory) < 64:
            return
        
        # Sample mini-batch from memory
        batch = random.sample(self.memory, min(64, len(self.memory)))
        
        self.network.train()
        total_loss = 0.0
        
        for exp_state, exp_action, exp_reward, exp_next_state, exp_done in batch:
            exp_state = exp_state.to(self.device)
            exp_next_state = exp_next_state.to(self.device)
            
            if len(exp_state.shape) == 2:
                exp_state = exp_state.unsqueeze(0)
                exp_next_state = exp_next_state.unsqueeze(0)
            
            # Forward pass
            action_probs, state_value, _ = self.network(exp_state)
            
            # Calculate target value
            with torch.no_grad():
                _, next_value, _ = self.network(exp_next_state)
                target_value = exp_reward + (0.99 * next_value * (1 - int(exp_done)))
            
            # Calculate losses
            value_loss = torch.nn.functional.mse_loss(state_value, target_value)
            
            # Policy gradient loss
            action_tensor = torch.tensor([exp_action], device=self.device)
            log_prob = torch.log(action_probs[0, action_tensor] + 1e-10)
            advantage = (target_value - state_value).detach()
            policy_loss = -(log_prob * advantage).mean()
            
            # Combined loss
            loss = value_loss + policy_loss
            
            # Backpropagation
            self.learner.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.learner.optimizer.step()
            
            total_loss += loss.item()
        
        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
        
        logger.debug(f"Learning complete. Loss: {total_loss:.4f}, Exploration: {self.exploration_rate:.3f}")
    
    def continuous_learning_loop(self, symbol: str, duration_hours: int = 24):
        """
        Continuous learning loop that runs indefinitely
        The agent learns from market data in real-time
        """
        logger.info(f"Starting continuous learning for {symbol} ({duration_hours} hours)")
        
        import asyncio
        
        async def learning_session():
            start_time = datetime.now()
            
            while True:
                # Check if we should stop
                elapsed = (datetime.now() - start_time).total_seconds() / 3600
                if elapsed >= duration_hours:
                    break
                
                # Fetch latest market data
                try:
                    market_data = await self.data_fetcher.fetch_market_data(
                        symbol,
                        timeframe='1m',
                        limit=100
                    )
                    
                    # Extract features
                    state = self._extract_features(market_data)
                    
                    # Make decision
                    action, confidence = self.decide_action(state)
                    
                    # Simulate reward (in real trading, this would be actual P&L)
                    reward = self._simulate_reward(market_data, action)
                    
                    # Get next state
                    await asyncio.sleep(60)  # Wait 1 minute
                    next_market_data = await self.data_fetcher.fetch_market_data(
                        symbol,
                        timeframe='1m',
                        limit=100
                    )
                    next_state = self._extract_features(next_market_data)
                    
                    # Learn from experience
                    self.learn_from_experience(state, action, reward, next_state, False)
                    
                    # Adapt learning rate based on performance
                    self.learner.adapt_learning_rate(reward)
                    
                    self.total_rewards += reward
                    self.episode_count += 1
                    
                    if self.episode_count % 100 == 0:
                        avg_reward = self.total_rewards / self.episode_count
                        logger.info(f"Episode {self.episode_count}, Avg Reward: {avg_reward:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in learning loop: {e}")
                    await asyncio.sleep(10)
        
        # Run the async learning session
        asyncio.run(learning_session())
    
    def _extract_features(self, market_data) -> torch.Tensor:
        """
        Extract features from market data for neural network input
        """
        # Simple feature extraction (can be made much more sophisticated)
        features = []
        
        # Price features
        if len(market_data) > 0:
            close_prices = market_data['close'].values
            
            # Returns
            returns = np.diff(close_prices) / close_prices[:-1]
            features.extend(returns[-20:].tolist() if len(returns) >= 20 else [0] * 20)
            
            # Moving averages
            ma_short = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else close_prices[-1]
            ma_long = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
            features.append(ma_short / ma_long - 1)
            
            # Volatility
            vol = np.std(returns[-20:]) if len(returns) >= 20 else 0
            features.append(vol)
            
            # Volume
            if 'volume' in market_data.columns:
                vol_ratio = market_data['volume'].iloc[-1] / market_data['volume'].mean()
                features.append(vol_ratio)
        
        # Pad to 100 features
        while len(features) < 100:
            features.append(0.0)
        
        # Convert to tensor with sequence dimension
        features_array = np.array(features[:100])
        features_tensor = torch.FloatTensor(features_array).unsqueeze(0)  # Add sequence dimension
        
        return features_tensor
    
    def _simulate_reward(self, market_data, action: int) -> float:
        """
        Simulate trading reward based on action and market movement
        In production, this would be replaced with actual P&L
        """
        if len(market_data) < 2:
            return 0.0
        
        # Get price change
        price_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2]
        
        # Reward based on action and price movement
        if action == 1:  # Buy
            reward = price_change * 100  # Profit if price went up
        elif action == 2:  # Sell
            reward = -price_change * 100  # Profit if price went down
        else:  # Hold
            reward = -0.01  # Small penalty for inaction
        
        return reward
    
    def save_model(self, path: str):
        """Save the trained model"""
        metadata = {
            'total_rewards': self.total_rewards,
            'episode_count': self.episode_count,
            'exploration_rate': self.exploration_rate,
            'timestamp': datetime.now().isoformat()
        }
        self.network.save_checkpoint(path, metadata)
    
    def load_model(self, path: str):
        """Load a trained model"""
        metadata = self.network.load_checkpoint(path)
        if metadata:
            self.total_rewards = metadata.get('total_rewards', 0.0)
            self.episode_count = metadata.get('episode_count', 0)
            self.exploration_rate = metadata.get('exploration_rate', self.config.EXPLORATION_RATE)
