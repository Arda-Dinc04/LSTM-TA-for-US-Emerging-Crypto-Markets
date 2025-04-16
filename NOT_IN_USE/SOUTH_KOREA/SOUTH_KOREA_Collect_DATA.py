import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# List of top South Korean stocks by sector
tickers = [
    # Technology/Electronics
    "005930.KS",  # Samsung Electronics
    "000660.KS",  # SK Hynix
    "066570.KS",  # LG Electronics
    "009150.KS",  # Samsung Electro-Mechanics
    "034220.KS",  # LG Display
    "006400.KS",  # Samsung SDI
    # Automotive
    "005380.KS",  # Hyundai Motor
    "000270.KS",  # Kia Corporation
    "012330.KS",  # Hyundai Mobis
    "011210.KS",  # Hyundai WIA
    # Energy/Chemical
    "051910.KS",  # LG Chem
    "096770.KS",  # SK Innovation
    "010950.KS",  # S-Oil
    "011170.KS",  # Lotte Chemical
    "078930.KS",  # GS
    # Steel/Metals
    "005490.KS",  # POSCO
    "004020.KS",  # Hyundai Steel
    "002380.KS",  # KG Dongbu Steel
    # Telecom/Internet
    "035420.KS",  # NAVER
    "035720.KS",  # Kakao
    "017670.KS",  # SK Telecom
    "030200.KS",  # KT Corporation
    "032640.KS",  # LG Uplus
    # Finance/Banking
    "055550.KS",  # Shinhan Financial Group
    "105560.KS",  # KB Financial Group
    "086790.KS",  # Hana Financial Group
    "316140.KS",  # Woori Financial Group
    "000810.KS",  # Samsung Fire & Marine Insurance
    # Retail/Distribution
    "023530.KS",  # Lotte Shopping
    "069960.KS",  # Hyundai Department Store
    "004170.KS",  # Shinsegae
    "139480.KS",  # Emart
    # Healthcare/Biotech
    "207940.KS",  # Samsung Biologics
    "068270.KS",  # Celltrion
    "128940.KS",  # Hanmi Pharmaceutical
    "326030.KS",  # SK Bioscience
    # Construction
    "000720.KS",  # Hyundai Engineering & Construction
    "028260.KS",  # Samsung C&T
    "047040.KS",  # Daewoo E&C
    "003490.KS",  # Dongbu Corporation
    # Food/Beverage
    "003550.KS",  # LG
    "097950.KS",  # CJ CheilJedang
    "004370.KS",  # Nongshim
    "026960.KS",  # Dongwon Industries
    # Entertainment/Media
    "035900.KS",  # JYP Entertainment
    "041510.KS",  # SM Entertainment
    "352820.KS",  # HYBE (formerly Big Hit Entertainment)
    "035760.KS",  # CJ ENM
    "035000.KS",  # KBS Media
    # Others
    "009540.KS",  # Hyundai Heavy Industries
    "010140.KS",  # Samsung Heavy Industries
    "018880.KS"   # Hanwha Ocean (former DSME)
]

# Time range - 5 years
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Approximately 5 years

# Create directory if it doesn't exist
data_dir = 'SOUTH_KOREA/SOUTH_KOREA_STOCK_DATA'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Fetch and save data
successful_downloads = []
failed_downloads = []

for ticker in tqdm(tickers, desc="Downloading South Korean stock data"):
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

print("\nAll South Korean stock data saved in the 'SOUTH_KOREA/SOUTH_KOREA_STOCK_DATA' folder!") 