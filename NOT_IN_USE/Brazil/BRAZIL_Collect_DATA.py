import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# List of major Brazilian stocks and Ibovespa index
brazil_tickers = [
    '^BVSP',     # Ibovespa Index
    'PETR4.SA',  # Petrobras
    'VALE3.SA',  # Vale
    'ITUB4.SA',  # Itaú Unibanco
    'BBDC4.SA',  # Bradesco
    'B3SA3.SA',  # B3
    'ABEV3.SA',  # Ambev
    'WEGE3.SA',  # WEG
    'RENT3.SA',  # Localiza
    'BBAS3.SA',  # Banco do Brasil
    'ITSA4.SA',  # Itaúsa
    'MGLU3.SA',  # Magazine Luiza
    'VIVO4.SA',  # Telefônica Brasil
    'LREN3.SA',  # Lojas Renner
    'RADL3.SA',  # Raia Drogasil
    'SUZB3.SA',  # Suzano
    'UGPA3.SA',  # Ultrapar
    'ENEV3.SA',  # Eneva
    'GGBR4.SA',  # Gerdau
    'RAIL3.SA',  # Rumo
    'JBSS3.SA',  # JBS
    'CSAN3.SA',  # Cosan
    'HYPE3.SA',  # Hypera
    'VIVT3.SA',  # Vivo
    'SANB11.SA', # Santander Brasil
    'BBSE3.SA',  # BB Seguridade
    'BRFS3.SA',  # BRF
    'CCRO3.SA',  # CCR
    'YDUQ3.SA',  # YDUQS
    'KLBN11.SA', # Klabin
]

# Define the time range - 5 years of data
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Approximately 5 years

# Create directory if it doesn't exist
data_dir = 'BRAZIL/BRAZIL_DATA'
os.makedirs(data_dir, exist_ok=True)

# Fetch and save data for each ticker
successful_downloads = []
failed_downloads = []

for ticker in tqdm(brazil_tickers, desc='Downloading Brazilian market data'):
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
print(f"Successfully downloaded: {len(successful_downloads)} Brazilian securities")
print(f"Failed to download: {len(failed_downloads)} Brazilian securities")
if failed_downloads:
    print("\nFailed downloads:")
    for ticker in failed_downloads:
        print(f"- {ticker}")

print("\nAll Brazilian market data saved in the 'BRAZIL/BRAZIL_DATA' folder!") 