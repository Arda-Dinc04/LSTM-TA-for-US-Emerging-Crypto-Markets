import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Technical Analysis imports
import talib
import numpy as np

# List of top Argentinian stocks by sector
tickers = [
    # Banking/Finance
    "BMA.BA", "GGAL.BA", "SUPV.BA", "BBAR.BA", "BYMA.BA", 
    # Energy/Oil & Gas
    "YPF.BA", "PAMP.BA", "TGNO4.BA", "TGSU2.BA", "METR.BA", "TRAN.BA",
    # Utilities
    "CEPU.BA", "CGPA2.BA", "EDN.BA", "TXAR.BA", "CAPX.BA",
    # Consumer/Retail
    "CRES.BA", "HAVA.BA", "LOMA.BA", "IRSA.BA", "CECO2.BA",
    # Agriculture/Food
    "MOLI.BA", "SAMI.BA", "INAG.BA", "SEMI.BA",
    # Telecom/Technology
    "TECO2.BA", "MIRG.BA", "CARC.BA", "AGRO.BA",
    # Steel/Materials
    "AUSO.BA", "FIPL.BA", "ALUA.BA", "CELU.BA",
    # Others
    "COME.BA", "RICH.BA", "BOLT.BA", "CTIO.BA", "GARO.BA",
    # ADRs
    "YPF", "GGAL", "BMA", "SUPV", "PAM", "TEO", "TGS", 
    "CEPU", "EDN", "CRESY", "LOMA", "IRCP", "IRS"
]

# Time range - 5 years
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Approximately 5 years

# Create directory if it doesn't exist
data_dir = 'ARGENTINA/ARGENTINA_STOCK_DATA'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Function to calculate technical indicators
def calculate_indicators(df):
    # Convert DataFrame columns to numpy arrays
    close = df['Close'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    
    # Calculate SMAs
    df['SMA_50'] = talib.SMA(close, timeperiod=50)
    df['SMA_200'] = talib.SMA(close, timeperiod=200)
    
    # Calculate EMAs
    df['EMA_9'] = talib.EMA(close, timeperiod=9)
    df['EMA_21'] = talib.EMA(close, timeperiod=21)
    df['EMA_50'] = talib.EMA(close, timeperiod=50)
    df['EMA_200'] = talib.EMA(close, timeperiod=200)
    
    # Calculate RSI
    df['RSI'] = talib.RSI(close, timeperiod=14)
    
    # Calculate MACD
    macd, signal, hist = talib.MACD(close, 
                                   fastperiod=12, 
                                   slowperiod=26, 
                                   signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    # Calculate Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, 
                                       timeperiod=20,
                                       nbdevup=2,
                                       nbdevdn=2,
                                       matype=0)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    
    # Calculate ADX
    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    df['ADX_Plus_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['ADX_Minus_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    
    return df

# Fetch and save data
successful_downloads = []
failed_downloads = []

for ticker in tqdm(tickers, desc="Downloading Argentinian stock data"):
    print(f"\nProcessing {ticker}...")
    try:
        # Add a small delay between requests to avoid rate limiting
        time.sleep(1)
        
        # Download data with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Download with auto_adjust=True to get adjusted prices
                df = yf.download(ticker, 
                                start=start_date.strftime('%Y-%m-%d'), 
                                end=end_date.strftime('%Y-%m-%d'), 
                                progress=False, 
                                auto_adjust=True)
                
                # Check if we have data
                if not df.empty and len(df) > 10:  # Ensure we have enough data
                    # Calculate technical indicators
                    df = calculate_indicators(df)
                    
                    # Create header rows like in the USA stock data
                    price_header = ['Price'] + list(df.columns)
                    ticker_header = ['Ticker'] + [ticker] * len(df.columns)
                    date_header = ['Date'] + [''] * len(df.columns)
                    
                    # Save the file with the same structure as USA stocks
                    with open(f'{data_dir}/{ticker.replace(".", "_")}.csv', 'w') as f:
                        # Write headers
                        f.write(','.join(price_header) + '\n')
                        f.write(','.join(ticker_header) + '\n')
                        f.write(','.join(date_header) + '\n')
                        
                        # Write data
                        df.to_csv(f, index=True, header=False)
                    
                    successful_downloads.append(ticker)
                    print(f"✓ Successfully downloaded and saved {ticker}")
                    break
                else:
                    print(f"⚠ Warning: No sufficient data found for {ticker}")
                    failed_downloads.append(ticker)
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"✗ Failed to download {ticker} after {max_retries} attempts: {str(e)}")
                    failed_downloads.append(ticker)
                else:
                    print(f"Attempt {attempt + 1} failed for {ticker}, retrying...")
                    time.sleep(2)
    except Exception as e:
        print(f"✗ Error processing {ticker}: {str(e)}")
        failed_downloads.append(ticker)

# Print summary
print("\nDownload Summary:")
print(f"Successfully downloaded: {len(successful_downloads)} stocks")
print(f"Failed to download: {len(failed_downloads)} stocks")
if failed_downloads:
    print("\nFailed downloads:")
    for ticker in failed_downloads:
        print(f"- {ticker}")

print("\nAll Argentinian stock data saved in the 'ARGENTINA/ARGENTINA_STOCK_DATA' folder!") 