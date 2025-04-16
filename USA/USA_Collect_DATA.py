import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Tickers from S&P 500, Dow Jones, and Nasdaq
usa_tickers = [
    # Major indices
    '^GSPC',    # S&P 500
    '^DJI',     # Dow Jones
    '^IXIC',    # Nasdaq Composite
    
    # Top S&P 500 stocks by market cap
    'AAPL',     # Apple
    'MSFT',     # Microsoft
    'AMZN',     # Amazon
    'NVDA',     # NVIDIA
    'GOOGL',    # Alphabet (Google) Class A
    'GOOG',     # Alphabet (Google) Class C
    'META',     # Meta Platforms (Facebook)
    'TSLA',     # Tesla
    'BRK-B',    # Berkshire Hathaway Class B
    'JPM',      # JPMorgan Chase
    'JNJ',      # Johnson & Johnson
    'V',        # Visa
    'PG',       # Procter & Gamble
    'UNH',      # UnitedHealth Group
    'HD',       # Home Depot
    'MA',       # Mastercard
    'BAC',      # Bank of America
    'XOM',      # Exxon Mobil
    'AVGO',     # Broadcom
    'COST',     # Costco
    
    # Additional Dow Jones components
    'CRM',      # Salesforce
    'MCD',      # McDonald's
    'GS',       # Goldman Sachs
    'CSCO',     # Cisco
    'WMT',      # Walmart
    'DIS',      # Disney
    'INTC',     # Intel
    
    # Additional tech/growth stocks
    'AMD',      # Advanced Micro Devices
    'NFLX',     # Netflix
    'ADBE',     # Adobe
    'PYPL',     # PayPal
    'QCOM',     # Qualcomm
    'BKNG',     # Booking Holdings
    'SBUX',     # Starbucks
    'TXN',      # Texas Instruments
    'ABNB',     # Airbnb
    'TMUS',     # T-Mobile
    'INTU',     # Intuit
    'AMAT',     # Applied Materials
    
    # Other sectors representation
    'PFE',      # Pfizer (Healthcare)
    'WFC',      # Wells Fargo (Financial)
    'CVX',      # Chevron (Energy)
    'CMCSA',    # Comcast (Communication)
    'KO',       # Coca-Cola (Consumer Staples)
    'PEP',      # PepsiCo (Consumer Staples)
    'T',        # AT&T (Communication)
    'VZ',       # Verizon (Communication)
    'ABT',      # Abbott Laboratories (Healthcare)
    'ORCL',     # Oracle (Technology)
    'TMO',      # Thermo Fisher Scientific (Healthcare)
    'MRK',      # Merck (Healthcare)
    'MS',       # Morgan Stanley (Financial)
    'F',        # Ford (Automotive)
    'GM',       # General Motors (Automotive)
    'BA',       # Boeing (Industrials)
    'CAT',      # Caterpillar (Industrials)
    'MMM',      # 3M (Industrials)
    'GE',       # General Electric (Industrials)
    'IBM',      # IBM (Technology)
]

# Define the time range - 5 years of data
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Approximately 5 years

# Create directory if it doesn't exist
data_dir = 'USA/USA_DATA'
os.makedirs(data_dir, exist_ok=True)

# Fetch and save data for each ticker
successful_downloads = []
failed_downloads = []

for ticker in tqdm(usa_tickers, desc='Downloading USA market data'):
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
                    with open(f'{data_dir}/{ticker.replace(".", "_").replace("-", "_")}.csv', 'w') as f:
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
print(f"Successfully downloaded: {len(successful_downloads)} USA securities")
print(f"Failed to download: {len(failed_downloads)} USA securities")
if failed_downloads:
    print("\nFailed downloads:")
    for ticker in failed_downloads:
        print(f"- {ticker}")

print("\nAll USA market data saved in the 'USA/USA_DATA' folder!") 