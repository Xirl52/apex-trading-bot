# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     APEX Trading Bot                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │         Main Entry Point               │
        │          (main.py)                     │
        └────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌───────────────────────┐   ┌──────────────────────┐
    │  Self-Learning Agent  │   │   Configuration      │
    │  (trading_agent.py)   │   │   (config.py)        │
    └───────────────────────┘   └──────────────────────┘
                │
                │
    ┌───────────┴───────────────────────────┐
    │                                       │
    ▼                                       ▼
┌──────────────────────┐         ┌─────────────────────┐
│  Neural Network      │         │  Data Fetcher       │
│  (neural_network.py) │         │  (data_fetcher.py)  │
└──────────────────────┘         └─────────────────────┘
    │                                       │
    │                                       │
    ▼                                       ▼
┌──────────────────────┐         ┌─────────────────────┐
│  - LSTM Layers       │         │  - Yahoo Finance    │
│  - Transformer       │         │  - CCXT Exchanges   │
│  - Attention         │         │  - Alpaca API       │
│  - Dense Layers      │         │  - News Sources     │
└──────────────────────┘         └─────────────────────┘
```

## Neural Network Architecture

```
Input Features (100 dimensions)
    │
    ▼
┌─────────────────────────────┐
│  LSTM Layer 1               │
│  (512 hidden units)         │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  LSTM Layer 2               │
│  (512 hidden units)         │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Transformer Encoder        │
│  (8 attention heads)        │
│  (2 layers)                 │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Dense Layer 1 (256)        │
│  + ReLU + Dropout           │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Dense Layer 2 (128)        │
│  + ReLU + Dropout           │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Dense Layer 3 (64)         │
│  + ReLU + Dropout           │
└─────────────────────────────┘
    │
    ├──────────────┬───────────────┐
    │              │               │
    ▼              ▼               ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│  HOLD  │   │   BUY    │   │   SELL   │
└────────┘   └──────────┘   └──────────┘
```

## Data Flow

```
1. Market Data Collection
   ┌────────────────────────────────────┐
   │  Multiple Sources (parallel)       │
   │  - Exchange APIs                   │
   │  - Market data providers           │
   │  - News & sentiment                │
   └────────────────────────────────────┘
                  │
                  ▼
2. Feature Extraction
   ┌────────────────────────────────────┐
   │  - Price returns                   │
   │  - Moving averages                 │
   │  - Technical indicators            │
   │  - Volatility metrics              │
   │  - Volume analysis                 │
   └────────────────────────────────────┘
                  │
                  ▼
3. Neural Network Processing
   ┌────────────────────────────────────┐
   │  - LSTM for temporal patterns      │
   │  - Transformer for context         │
   │  - Attention mechanisms            │
   └────────────────────────────────────┘
                  │
                  ▼
4. Decision Making
   ┌────────────────────────────────────┐
   │  - Action selection                │
   │  - Confidence calculation          │
   │  - Risk assessment                 │
   └────────────────────────────────────┘
                  │
                  ▼
5. Learning & Adaptation
   ┌────────────────────────────────────┐
   │  - Experience replay               │
   │  - Backpropagation                 │
   │  - Learning rate adaptation        │
   │  - Model checkpointing             │
   └────────────────────────────────────┘
```

## Learning Loop

```
┌────────────────────────────────────────────────┐
│  Continuous Learning Cycle                     │
│                                                │
│  1. Observe State                              │
│     └─> Extract features from market data      │
│                                                │
│  2. Take Action                                │
│     └─> Epsilon-greedy policy                  │
│         (Explore vs Exploit)                   │
│                                                │
│  3. Receive Reward                             │
│     └─> Based on profit/loss                   │
│                                                │
│  4. Store Experience                           │
│     └─> Add to replay memory                   │
│                                                │
│  5. Learn from Batch                           │
│     └─> Sample from memory                     │
│     └─> Update network weights                 │
│                                                │
│  6. Adapt Learning Rate                        │
│     └─> Based on performance                   │
│                                                │
│  └─> Repeat indefinitely                       │
└────────────────────────────────────────────────┘
```

## Key Components

### 1. UnrestrictedNeuralNetwork
- Hybrid architecture (LSTM + Transformer + Dense)
- Multi-head attention for context understanding
- Dual output heads (actions + value)
- Supports GPU acceleration

### 2. SelfLearningAgent
- Reinforcement learning (A2C algorithm)
- Experience replay memory
- Adaptive exploration rate
- Continuous learning capability

### 3. UniversalDataFetcher
- Multi-source data aggregation
- Async data fetching
- Fallback mechanisms
- Caching and optimization

### 4. AdaptiveLearner
- Dynamic learning rate adjustment
- Performance-based adaptation
- Gradient clipping
- Optimizer management

## Unrestricted Features

The bot is "unrestricted" in several ways:

1. **Data Access**: Can connect to any API or data source
2. **Learning**: No artificial limits on training duration
3. **Strategy**: Free to explore any trading strategy
4. **Adaptation**: Continuously evolves without predefined rules
5. **Scalability**: Can trade unlimited symbols simultaneously

## Safety Mechanisms

Despite being unrestricted, safety is ensured through:

1. **Risk Management**: Position size limits, stop losses
2. **Paper Trading**: Default mode is simulation
3. **Monitoring**: Comprehensive logging and alerts
4. **Checkpointing**: Regular model saves
5. **Validation**: Data quality checks
