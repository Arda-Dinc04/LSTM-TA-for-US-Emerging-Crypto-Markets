import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# List of major cryptocurrencies
crypto_tickers = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum
    'BNB-USD',   # Binance Coin
    'SOL-USD',   # Solana
    'XRP-USD',   # XRP (Ripple)
    'ADA-USD',   # Cardano
    'DOGE-USD',  # Dogecoin
    'AVAX-USD',  # Avalanche
    'MATIC-USD', # Polygon
    'DOT-USD',   # Polkadot
    'LINK-USD',  # Chainlink
    'SHIB-USD',  # Shiba Inu
    'LTC-USD',   # Litecoin
    'UNI-USD',   # Uniswap
    'ATOM-USD',  # Cosmos
    'XLM-USD',   # Stellar
    'ALGO-USD',  # Algorand
    'FIL-USD',   # Filecoin
    'AAVE-USD',  # Aave
    'BCH-USD',   # Bitcoin Cash
    'NEAR-USD',  # NEAR Protocol
    'VET-USD',   # VeChain
    'ETC-USD',   # Ethereum Classic
    'HBAR-USD',  # Hedera
    'EGLD-USD',  # MultiversX (Elrond)
    'FTM-USD',   # Fantom
    'SAND-USD',  # The Sandbox
    'MANA-USD',  # Decentraland
    'AXS-USD',   # Axie Infinity
    'XTZ-USD',   # Tezos
]

# Define the time range - 5 years of data
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Approximately 5 years

# Create directory if it doesn't exist
data_dir = 'CRYPTO/CRYPTO_DATA'
os.makedirs(data_dir, exist_ok=True)

# Fetch and save data for each ticker
successful_downloads = []
failed_downloads = []

for ticker in tqdm(crypto_tickers, desc='Downloading crypto data'):
    try:
        # Add a small delay between requests to avoid rate limiting
        time.sleep(1)
        
        # Download data with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, 
                               start=start_date.strftime('%Y-%m-%d'), 
                               end=end_date.strftime('%Y-%m-%d'), 
                               progress=False,
                               auto_adjust=True)
                
                # Check if we have data
                if not df.empty and len(df) > 10:  # Ensure we have enough data
                    # Convert column names to strings to handle potential tuples
                    column_names = [str(col) for col in df.columns]
                    
                    # Create header rows
                    price_header = ['Price'] + column_names
                    ticker_header = ['Ticker'] + [ticker] * len(column_names)
                    date_header = ['Date'] + [''] * len(column_names)
                    
                    # Save the file
                    with open(f'{data_dir}/{ticker.replace("-", "_")}.csv', 'w') as f:
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
print(f"Successfully downloaded: {len(successful_downloads)} cryptocurrencies")
print(f"Failed to download: {len(failed_downloads)} cryptocurrencies")
if failed_downloads:
    print("\nFailed downloads:")
    for ticker in failed_downloads:
        print(f"- {ticker}")

print("\nAll crypto data saved in the 'CRYPTO/CRYPTO_DATA' folder!") 