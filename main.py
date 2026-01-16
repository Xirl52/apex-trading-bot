"""
Main entry point for APEX Trading Bot
Command-line interface for running the self-learning neural network
"""

import argparse
import asyncio
import logging
from typing import Optional
import sys

from apex_bot.config import Config
from apex_bot.trading_agent import SelfLearningAgent
from apex_bot.data_fetcher import UniversalDataFetcher


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('apex_bot.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the APEX trading bot"""
    parser = argparse.ArgumentParser(
        description='APEX Trading Bot - Unrestricted Self-Learning Neural Network'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'trade', 'analyze', 'demo'],
        help='Operation mode: train (learn from data), trade (live trading), analyze (performance), demo (demonstration)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USD',
        help='Trading symbol (e.g., BTC/USD, AAPL, ETH/USDT)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=24,
        help='Duration in hours for training/trading'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to load/save model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration file'
    )
    
    args = parser.parse_args()
    
    # Display banner
    print_banner()
    
    # Validate configuration
    try:
        Config.validate()
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Initialize the agent
    logger.info("Initializing self-learning agent...")
    agent = SelfLearningAgent(Config)
    
    # Load existing model if specified
    if args.model_path:
        try:
            agent.load_model(args.model_path)
            logger.info(f"Loaded model from {args.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    # Execute based on mode
    try:
        if args.mode == 'train':
            logger.info(f"Starting training mode for {args.symbol}")
            agent.continuous_learning_loop(args.symbol, args.duration)
            
            # Save model after training
            if args.model_path:
                agent.save_model(args.model_path)
                logger.info(f"Model saved to {args.model_path}")
        
        elif args.mode == 'trade':
            logger.info(f"Starting live trading mode for {args.symbol}")
            if not Config.ENABLE_LIVE_TRADING:
                logger.warning("Live trading is disabled in configuration!")
                logger.info("Running in paper trading mode instead")
            run_trading_mode(agent, args.symbol, args.duration)
        
        elif args.mode == 'analyze':
            logger.info("Analyzing agent performance...")
            analyze_performance(agent)
        
        elif args.mode == 'demo':
            logger.info("Running demonstration mode...")
            run_demo(agent, args.symbol)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Shutting down APEX Trading Bot")


def print_banner():
    """Print the APEX bot banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           APEX TRADING BOT - Neural Network System           ║
    ║                                                               ║
    ║     🧠 Unrestricted Self-Learning AI Trading Agent 🧠        ║
    ║                                                               ║
    ║  Features:                                                    ║
    ║    • Hybrid Neural Network (LSTM + Transformer)               ║
    ║    • Continuous Self-Learning                                 ║
    ║    • Unrestricted Data Access                                 ║
    ║    • Multi-Source Market Data                                 ║
    ║    • Adaptive Learning Rate                                   ║
    ║    • Reinforcement Learning                                   ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_trading_mode(agent: SelfLearningAgent, symbol: str, duration: int):
    """Run the bot in trading mode"""
    logger.info(f"Trading {symbol} for {duration} hours")
    
    # In a real implementation, this would connect to exchanges
    # and execute actual trades based on agent decisions
    agent.continuous_learning_loop(symbol, duration)


def analyze_performance(agent: SelfLearningAgent):
    """Analyze and display agent performance metrics"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Total Episodes: {agent.episode_count}")
    print(f"Total Rewards: {agent.total_rewards:.2f}")
    if agent.episode_count > 0:
        print(f"Average Reward: {agent.total_rewards / agent.episode_count:.4f}")
    print(f"Exploration Rate: {agent.exploration_rate:.3f}")
    print(f"Learning Rate: {agent.learner.get_current_lr():.6f}")
    print(f"Memory Size: {len(agent.memory)}")
    print("="*60 + "\n")


def run_demo(agent: SelfLearningAgent, symbol: str):
    """Run a quick demonstration of the bot capabilities"""
    logger.info("Demonstration Mode - Showing bot capabilities")
    
    print("\n🎯 Fetching market data...")
    
    async def demo():
        # Fetch data
        data = await agent.data_fetcher.fetch_market_data(symbol, limit=100)
        print(f"✓ Fetched {len(data)} data points for {symbol}")
        print(f"  Latest price: ${data['close'].iloc[-1]:.2f}")
        
        # Make a prediction
        state = agent._extract_features(data)
        action, confidence = agent.decide_action(state, deterministic=True)
        
        actions_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        print(f"\n🤖 AI Decision: {actions_map[action]}")
        print(f"   Confidence: {confidence*100:.1f}%")
        
        # Show network info
        print(f"\n🧠 Network Information:")
        print(f"   Type: {agent.network.network_type}")
        print(f"   Parameters: {agent.network._count_parameters():,}")
        print(f"   Device: {agent.device}")
        
        # Show configuration
        print(f"\n⚙️  Configuration:")
        print(f"   Self-Learning: {Config.ENABLE_SELF_LEARNING}")
        print(f"   Unrestricted Mode: {Config.UNRESTRICTED_MODE}")
        print(f"   Data Sources: {', '.join(Config.DATA_SOURCES)}")
        
        await agent.data_fetcher.close()
    
    asyncio.run(demo())
    print("\n✓ Demonstration complete!\n")


if __name__ == '__main__':
    main()
