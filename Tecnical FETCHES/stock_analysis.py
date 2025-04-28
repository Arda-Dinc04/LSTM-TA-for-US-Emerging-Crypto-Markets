import yfinance as yf
import pandas as pd
import numpy as np
import ta

def fetch_stock_data(ticker="AAPL", period="5y", interval="1d"):
    # Fetch stock data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
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

def main():
    # Fetch and process data
    df = fetch_stock_data()
    
    # Save to CSV
    output_file = 'AAPL_5y_technical_analysis.csv'
    df.to_csv(output_file)
    print(f"Data has been saved to {output_file}")

if __name__ == "__main__":
    main() 