"""
Setup configuration for APEX Trading Bot
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="apex-trading-bot",
    version="1.0.0",
    author="Xirl52",
    description="Advanced Self-Learning Neural Network Trading System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Xirl52/apex-trading-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "websockets>=11.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=1.0.0",
        "ta>=0.11.0",
        "matplotlib>=3.7.0",
        "gym>=0.26.0",
        "stable-baselines3>=2.0.0",
        "ccxt>=4.0.0",
        "yfinance>=0.2.0",
        "alpaca-py>=0.8.0",
    ],
    entry_points={
        "console_scripts": [
            "apex-bot=main:main",
        ],
    },
)
