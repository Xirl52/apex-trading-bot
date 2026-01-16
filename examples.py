"""
APEX Trading Bot - Examples and Tutorials

This file contains examples showing how to use the bot in different scenarios
"""

import asyncio
from apex_bot.config import Config
from apex_bot.neural_network import UnrestrictedNeuralNetwork, AdaptiveLearner
from apex_bot.trading_agent import SelfLearningAgent
from apex_bot.data_fetcher import UniversalDataFetcher
import torch


def example_1_basic_usage():
    """Example 1: Basic bot initialization and single prediction"""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Create agent
    agent = SelfLearningAgent(Config)
    
    # Create a sample state (100 features)
    sample_state = torch.randn(1, 100)  # batch_size=1, seq_len=1, features=100
    
    # Make a decision
    action, confidence = agent.decide_action(sample_state, deterministic=True)
    
    actions = ["HOLD", "BUY", "SELL"]
    print(f"Decision: {actions[action]}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Network has {agent.network._count_parameters():,} parameters")


def example_2_data_fetching():
    """Example 2: Fetch market data from multiple sources"""
    print("\n" + "="*60)
    print("Example 2: Data Fetching")
    print("="*60)
    
    async def fetch_data():
        data_fetcher = UniversalDataFetcher(Config)
        
        # Fetch Bitcoin data
        btc_data = await data_fetcher.fetch_market_data('BTC/USD', limit=100)
        print(f"Fetched {len(btc_data)} BTC data points")
        print(f"Latest BTC price: ${btc_data['close'].iloc[-1]:.2f}")
        print(f"Price range: ${btc_data['low'].min():.2f} - ${btc_data['high'].max():.2f}")
        
        await data_fetcher.close()
    
    asyncio.run(fetch_data())


def example_3_training_loop():
    """Example 3: Short training loop demonstration"""
    print("\n" + "="*60)
    print("Example 3: Training Loop (1 hour)")
    print("="*60)
    
    agent = SelfLearningAgent(Config)
    
    print("Starting training for 1 hour on BTC/USD...")
    print("(This will fetch data, make decisions, and learn)")
    
    # Train for 1 hour
    # Uncomment to actually run:
    # agent.continuous_learning_loop('BTC/USD', duration_hours=1)
    
    print("Training would run here. Skipped in example.")
    print("In real usage, the agent would:")
    print("  1. Fetch market data every minute")
    print("  2. Make trading decisions")
    print("  3. Learn from outcomes")
    print("  4. Adapt learning rate based on performance")


def example_4_custom_network():
    """Example 4: Create a custom neural network configuration"""
    print("\n" + "="*60)
    print("Example 4: Custom Network Configuration")
    print("="*60)
    
    # Create a custom network with different architecture
    custom_network = UnrestrictedNeuralNetwork(
        input_size=100,
        hidden_layers=[1024, 512, 256, 128],  # Larger network
        output_size=3,
        dropout_rate=0.3,
        network_type="transformer"  # Pure transformer
    )
    
    print(f"Network type: {custom_network.network_type}")
    print(f"Parameters: {custom_network._count_parameters():,}")
    print(f"Hidden layers: [1024, 512, 256, 128]")
    
    # Test forward pass
    test_input = torch.randn(1, 10, 100)  # batch, sequence, features
    action_probs, value, _ = custom_network(test_input)
    
    print(f"Output shape - Actions: {action_probs.shape}, Value: {value.shape}")


def example_5_save_load_model():
    """Example 5: Save and load trained model"""
    print("\n" + "="*60)
    print("Example 5: Model Persistence")
    print("="*60)
    
    # Create and train agent
    agent = SelfLearningAgent(Config)
    
    # Simulate some training by updating episode count
    agent.episode_count = 1000
    agent.total_rewards = 543.21
    
    # Save model
    model_path = '/tmp/apex_model.pth'
    agent.save_model(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Create new agent and load model
    new_agent = SelfLearningAgent(Config)
    new_agent.load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    print(f"  Episodes: {new_agent.episode_count}")
    print(f"  Total rewards: {new_agent.total_rewards:.2f}")


def example_6_adaptive_learning():
    """Example 6: Demonstrate adaptive learning rate"""
    print("\n" + "="*60)
    print("Example 6: Adaptive Learning Rate")
    print("="*60)
    
    network = UnrestrictedNeuralNetwork(
        input_size=100,
        hidden_layers=[256, 128, 64],
        output_size=3
    )
    
    learner = AdaptiveLearner(network, initial_lr=0.001)
    
    print(f"Initial learning rate: {learner.get_current_lr():.6f}")
    
    # Simulate performance improvements
    for i in range(5):
        performance = 0.5 + i * 0.1  # Improving performance
        learner.adapt_learning_rate(performance)
        print(f"After performance {performance:.2f}: LR = {learner.get_current_lr():.6f}")


def example_7_multi_symbol_trading():
    """Example 7: Trade multiple symbols simultaneously"""
    print("\n" + "="*60)
    print("Example 7: Multi-Symbol Trading")
    print("="*60)
    
    symbols = ['BTC/USD', 'ETH/USD', 'AAPL', 'GOOGL']
    
    print("In production, you would run multiple agents:")
    for symbol in symbols:
        print(f"  Agent for {symbol}")
    
    print("\nEach agent would:")
    print("  - Have its own neural network")
    print("  - Learn independently")
    print("  - Make symbol-specific decisions")
    print("  - Can share learned patterns (transfer learning)")


def example_8_unrestricted_features():
    """Example 8: Showcase unrestricted/smart features"""
    print("\n" + "="*60)
    print("Example 8: Unrestricted AI Features")
    print("="*60)
    
    print("\n🧠 Intelligence Features:")
    print("  ✓ Hybrid Neural Network (LSTM + Transformer)")
    print("  ✓ Self-attention mechanisms")
    print("  ✓ Experience replay learning")
    print("  ✓ Adaptive learning rate")
    
    print("\n🌐 Unrestricted Data Access:")
    print("  ✓ Multiple exchange APIs (CCXT)")
    print("  ✓ Traditional finance data (Yahoo, Alpaca)")
    print("  ✓ News sentiment analysis")
    print("  ✓ Social media sentiment")
    print("  ✓ Alternative data sources")
    
    print("\n🎯 Self-Learning Capabilities:")
    print("  ✓ Continuous learning from market data")
    print("  ✓ Reinforcement learning (A2C algorithm)")
    print("  ✓ Exploration vs exploitation balance")
    print("  ✓ Automatic strategy adaptation")
    
    print("\n⚡ Advanced Features:")
    print("  ✓ GPU acceleration support")
    print("  ✓ Model checkpointing")
    print("  ✓ Performance analytics")
    print("  ✓ Risk management")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print(" "*15 + "APEX TRADING BOT - EXAMPLES")
    print("="*70)
    
    try:
        example_1_basic_usage()
        example_2_data_fetching()
        example_3_training_loop()
        example_4_custom_network()
        example_5_save_load_model()
        example_6_adaptive_learning()
        example_7_multi_symbol_trading()
        example_8_unrestricted_features()
        
        print("\n" + "="*70)
        print(" "*20 + "All examples completed!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")


if __name__ == '__main__':
    run_all_examples()
