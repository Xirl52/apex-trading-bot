# Quick Start Guide

Get started with APEX Trading Bot in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Xirl52/apex-trading-bot.git
cd apex-trading-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will install PyTorch, TensorFlow, and other ML libraries. Installation may take 5-10 minutes.

### 3. Configure the Bot (Optional)

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` to add your API keys (optional for demo):

```bash
# For live data access (optional)
ALPHA_VANTAGE_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_key_here

# Keep this false for safety!
ENABLE_LIVE_TRADING=false
```

## Running Your First Bot

### Demo Mode (Recommended First Step)

Run a quick demonstration:

```bash
python main.py --mode demo --symbol BTC/USD
```

This will:
- Show the bot's capabilities
- Fetch sample market data
- Make a prediction
- Display network information

### Training Mode

Train the neural network on historical data:

```bash
python main.py --mode train --symbol BTC/USD --duration 1 --model-path /tmp/btc_model.pth
```

Parameters:
- `--symbol`: Trading symbol (BTC/USD, ETH/USD, AAPL, etc.)
- `--duration`: Training duration in hours
- `--model-path`: Where to save the trained model

### View Examples

See detailed examples of all features:

```bash
python examples.py
```

### Analyze Performance

After training, analyze the bot's performance:

```bash
python main.py --mode analyze --model-path /tmp/btc_model.pth
```

## Understanding the Output

When you run the bot, you'll see:

```
╔═══════════════════════════════════════════════════════════════╗
║           APEX TRADING BOT - Neural Network System           ║
║     🧠 Unrestricted Self-Learning AI Trading Agent 🧠        ║
╚═══════════════════════════════════════════════════════════════╝

🎯 Fetching market data...
✓ Fetched 100 data points for BTC/USD
  Latest price: $45,123.45

🤖 AI Decision: BUY
   Confidence: 67.3%

🧠 Network Information:
   Type: hybrid
   Parameters: 1,234,567
   Device: cuda
```

## Next Steps

1. **Read the Documentation**
   - [README.md](README.md) - Overview and features
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
   - [examples.py](examples.py) - Code examples

2. **Experiment with Different Symbols**
   ```bash
   python main.py --mode train --symbol ETH/USD --duration 2
   python main.py --mode train --symbol AAPL --duration 2
   ```

3. **Customize the Neural Network**
   
   Edit `.env` to change network configuration:
   ```bash
   NETWORK_TYPE=transformer  # Options: hybrid, lstm, transformer
   LEARNING_RATE=0.0001
   EXPLORATION_RATE=0.5
   ```

4. **Advanced: Multi-Symbol Trading**
   
   Run multiple instances for different symbols:
   ```bash
   python main.py --mode train --symbol BTC/USD --duration 24 &
   python main.py --mode train --symbol ETH/USD --duration 24 &
   python main.py --mode train --symbol AAPL --duration 24 &
   ```

## Common Issues

### Issue: ModuleNotFoundError

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: CUDA out of memory

**Solution:** The bot will automatically fall back to CPU. To force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
python main.py --mode demo
```

### Issue: Data fetching fails

**Solution:** The bot includes fallback to synthetic data for testing. For real data, add API keys to `.env`

## Safety Reminders

⚠️ **Important Safety Information:**

1. **Paper Trading by Default**: The bot runs in paper trading mode by default
2. **Never Enable Live Trading**: Unless you fully understand the risks
3. **Educational Purpose**: This bot is for learning and research only
4. **No Financial Advice**: Always do your own research
5. **Test Thoroughly**: Test extensively before any real-world use

## Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check existing issues or create a new one
- **Contributions**: See CONTRIBUTING.md (if available)

## What's Next?

Now that you're set up, explore:

- Training on different timeframes
- Customizing the neural network architecture
- Adding new data sources
- Implementing custom features
- Understanding the self-learning mechanism

Happy Trading (Responsibly)! 🚀
