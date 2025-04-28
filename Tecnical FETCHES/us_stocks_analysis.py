import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
from datetime import datetime

# Top 50 US stocks from various industries with their sector classification
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

def fetch_stock_data(ticker, period="5y", interval="1d"):
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            print(f"No data available for {ticker}")
            return None
        
        # Calculate Simple Moving Averages
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # Calculate MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Calculate ADX
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx.adx()
        
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()
        
        return df
    
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

def main():
    # Create US Stocks directory if it doesn't exist
    output_dir = "US Stocks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each stock
    successful_stocks = []
    failed_stocks = []
    
    for ticker, industry in STOCKS.items():
        print(f"Processing {ticker} ({industry})...")
        df = fetch_stock_data(ticker)
        
        if df is not None:
            output_file = os.path.join(output_dir, f"{ticker}_5y_technical_analysis.csv")
            df.to_csv(output_file)
            successful_stocks.append((ticker, industry))
            print(f"Data saved to {output_file}")
        else:
            failed_stocks.append((ticker, industry))
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Successfully processed: {len(successful_stocks)} stocks")
    print(f"Failed to process: {len(failed_stocks)} stocks")
    
    if failed_stocks:
        print("\nFailed stocks:")
        for ticker, industry in failed_stocks:
            print(f"- {ticker} ({industry})")

if __name__ == "__main__":
    main() 