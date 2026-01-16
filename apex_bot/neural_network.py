"""
Unrestricted Neural Network - Advanced AI architecture
Self-learning and adaptive neural network for trading decisions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UnrestrictedNeuralNetwork(nn.Module):
    """
    Advanced neural network with unrestricted learning capabilities
    Combines multiple architectures for maximum intelligence:
    - LSTM for temporal patterns
    - Transformer for attention mechanisms
    - Dense layers for complex decision making
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        dropout_rate: float = 0.2,
        network_type: str = "hybrid"
    ):
        super(UnrestrictedNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.network_type = network_type
        
        # LSTM layers for temporal pattern recognition
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layers[0],
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Transformer encoder for attention mechanisms
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_layers[0],
            nhead=8,
            dim_feedforward=hidden_layers[0] * 2,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Dense layers for decision making
        self.dense_layers = nn.ModuleList()
        prev_size = hidden_layers[0]
        
        for hidden_size in hidden_layers[1:]:
            self.dense_layers.append(nn.Linear(prev_size, hidden_size))
            self.dense_layers.append(nn.ReLU())
            self.dense_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layers
        self.action_head = nn.Linear(prev_size, output_size)  # Trading actions
        self.value_head = nn.Linear(prev_size, 1)  # State value estimation
        
        # Attention weights for interpretability
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_layers[0],
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        logger.info(f"Initialized {network_type} neural network with {self._count_parameters()} parameters")
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch, sequence, features)
            hidden_state: Optional LSTM hidden state
            
        Returns:
            action_probs: Action probabilities
            state_value: Estimated state value
            new_hidden_state: Updated LSTM hidden state
        """
        batch_size = x.size(0)
        
        # LSTM processing for temporal patterns
        if self.network_type in ["hybrid", "lstm"]:
            lstm_out, new_hidden = self.lstm(x, hidden_state)
            x = lstm_out
        else:
            new_hidden = None
        
        # Transformer processing for attention
        if self.network_type in ["hybrid", "transformer"]:
            # Self-attention mechanism
            attended, attention_weights = self.attention(x, x, x)
            x = x + attended  # Residual connection
            x = self.transformer(x)
        
        # Take the last timestep
        x = x[:, -1, :]
        
        # Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        # Output heads
        action_logits = self.action_head(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        state_value = self.value_head(x)
        
        return action_probs, state_value, new_hidden
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'network_type': self.network_type,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('metadata', {})


class AdaptiveLearner:
    """
    Adaptive learning module that adjusts learning rate and strategy
    based on performance and market conditions
    """
    
    def __init__(
        self,
        model: UnrestrictedNeuralNetwork,
        initial_lr: float = 0.001,
        min_lr: float = 0.00001,
        max_lr: float = 0.01
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=min_lr
        )
        
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.performance_history: List[float] = []
        
    def adapt_learning_rate(self, performance: float):
        """
        Dynamically adapt learning rate based on performance
        This makes the network "smarter" by optimizing its learning
        """
        self.performance_history.append(performance)
        
        # Use scheduler for automatic adjustment
        self.scheduler.step(performance)
        
        # Additional adaptive logic
        if len(self.performance_history) > 20:
            recent_performance = np.mean(self.performance_history[-20:])
            older_performance = np.mean(self.performance_history[-40:-20])
            
            # If improving, can afford to explore more
            if recent_performance > older_performance:
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = min(current_lr * 1.1, self.max_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
        logger.debug(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
