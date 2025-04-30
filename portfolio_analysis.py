import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Stock sector mapping
STOCKS = {
    # Technology
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'META': 'Technology',
    'NVDA': 'Technology',
    'AVGO': 'Technology',
    'ORCL': 'Technology',
    'CRM': 'Technology',
    'AMD': 'Technology',
    'INTC': 'Technology',
    
    # Healthcare
    'JNJ': 'Healthcare',
    'UNH': 'Healthcare',
    'LLY': 'Healthcare',
    'PFE': 'Healthcare',
    'ABT': 'Healthcare',
    
    # Financial Services
    'JPM': 'Financial Services',
    'BAC': 'Financial Services',
    'V': 'Financial Services',
    'MA': 'Financial Services',
    'WFC': 'Financial Services',
    
    # Consumer Goods
    'PG': 'Consumer Goods',
    'KO': 'Consumer Goods',
    'PEP': 'Consumer Goods',
    'COST': 'Consumer Goods',
    'WMT': 'Consumer Goods',
    
    # Industrial
    'CAT': 'Industrial',
    'BA': 'Industrial',
    'HON': 'Industrial',
    'UPS': 'Industrial',
    'GE': 'Industrial',
    
    # Energy
    'XOM': 'Energy',
    'CVX': 'Energy',
    'COP': 'Energy',
    'SLB': 'Energy',
    'EOG': 'Energy',
    
    # Telecommunications
    'T': 'Telecommunications',
    'VZ': 'Telecommunications',
    'TMUS': 'Telecommunications',
    
    # Real Estate
    'AMT': 'Real Estate',
    'PLD': 'Real Estate',
    'CCI': 'Real Estate',
    
    # Materials
    'LIN': 'Materials',
    'APD': 'Materials',
    'FCX': 'Materials',
    
    # Utilities
    'NEE': 'Utilities',
    'DUK': 'Utilities',
    'SO': 'Utilities',
    
    # Entertainment & Media
    'NFLX': 'Entertainment & Media',
    'DIS': 'Entertainment & Media',
    'CMCSA': 'Entertainment & Media',
    
    # E-commerce
    'AMZN': 'E-commerce',
    'EBAY': 'E-commerce',
    'ETSY': 'E-commerce'
}

# Cryptocurrency categories based on typical market cap and use case
CRYPTO_CATEGORIES = {
    'Large Cap (>$10B)': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'USDT-USD', 'USDC-USD'],
    'DeFi': ['AAVE-USD', 'UNI-USD', 'COMP-USD', 'MKR-USD', 'SNX-USD', 'CRV-USD', 'CAKE-USD'],
    'Smart Contract Platforms': ['SOL-USD', 'ADA-USD', 'DOT-USD', 'AVAX-USD', 'NEAR-USD', 'FTM-USD', 'ALGO-USD'],
    'Web3 & Gaming': ['MANA-USD', 'SAND-USD', 'AXS-USD', 'GALA-USD'],
    'Infrastructure': ['LINK-USD', 'GRT-USD', 'MATIC-USD', 'FIL-USD', 'HBAR-USD'],
    'Exchange Tokens': ['FTT-USD', 'OKB-USD'],
    'Stablecoins': ['DAI-USD', 'USDT-USD', 'USDC-USD'],
    'Other Altcoins': ['LTC-USD', 'DOGE-USD', 'XLM-USD', 'EOS-USD', 'TRX-USD', 'ZEC-USD', 'DASH-USD']
}

def analyze_stock_distribution():
    """Analyze and visualize stock distribution by industry."""
    # Get list of stock files
    stock_files = list(Path('CRYPTO_STOCKS').glob('*_5y_technical_analysis.csv'))
    
    # Count stocks by sector
    sector_counts = {}
    for file in stock_files:
        ticker = file.name.split('_')[0]
        if ticker in STOCKS:
            sector = STOCKS[ticker]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    # Create pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%')
    plt.title('Stock Distribution by Industry', pad=20)
    plt.axis('equal')
    plt.savefig('stock_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return sector_counts

def analyze_crypto_distribution():
    """Analyze and visualize cryptocurrency distribution by category."""
    # Get list of crypto files
    crypto_files = list(Path('Crypto').glob('*.csv'))
    crypto_symbols = [f.stem for f in crypto_files]
    
    # Count cryptos by category
    category_counts = {}
    for category, symbols in CRYPTO_CATEGORIES.items():
        count = sum(1 for symbol in crypto_symbols if symbol in symbols)
        if count > 0:
            category_counts[category] = count
    
    # Create pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
    plt.title('Cryptocurrency Distribution by Category', pad=20)
    plt.axis('equal')
    plt.savefig('crypto_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return category_counts

def main():
    # Set style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Analyze stocks
    sector_counts = analyze_stock_distribution()
    print("\nStock Distribution by Industry:")
    for sector, count in sector_counts.items():
        print(f"{sector}: {count} stocks ({count/sum(sector_counts.values())*100:.1f}%)")
    
    # Analyze cryptocurrencies
    category_counts = analyze_crypto_distribution()
    print("\nCryptocurrency Distribution by Category:")
    for category, count in category_counts.items():
        print(f"{category}: {count} tokens ({count/sum(category_counts.values())*100:.1f}%)")

if __name__ == "__main__":
    main() 