import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# List of top Greek stocks by sector
tickers = [
    # Banking/Finance
    "ETE.AT",    # National Bank of Greece
    "ALPHA.AT",  # Alpha Bank
    "EUROB.AT",  # Eurobank
    "TPEIR.AT",  # Piraeus Bank
    # Energy/Utilities
    "PPC.AT",    # Public Power Corporation
    "ELPE.AT",   # Hellenic Petroleum
    "ELLAKTOR.AT", # Ellaktor
    "MYTIL.AT",  # Mytilineos
    "ADMIE.AT",  # IPTO Holding
    "TENERGY.AT", # Terna Energy
    # Telecom/Technology
    "HTO.AT",    # Hellenic Telecommunications
    "ENTERSOFT.AT", # Entersoft
    "PROF.AT",   # Profile Systems
    "IDEAL.AT",  # Ideal Holdings
    "LOGISMOS.AT", # Logismos
    # Retail/Consumer
    "INKAT.AT",  # Intracom
    "LAMDA.AT",  # Lamda Development
    "PLAISIO.AT", # Plaisio
    "FOURLIS.AT", # Fourlis Holdings
    "GEKTERNA.AT", # GEK Terna
    # Healthcare
    "IATR.AT",   # Athens Medical Group
    "LAVIPHARM.AT", # Lavipharm
    # Food/Beverage
    "HE.AT",     # Hellenic Exchanges
    "FRIGO.AT",  # Frigoglass
    "NIKAS.AT",  # P.G. Nikas
    "CRETA.AT",  # Creta Farms
    "KRI.AT",    # Kri-Kri Milk Industry
    # Industrial/Manufacturing
    "TITAN.AT",  # Titan Cement
    "ELTON.AT",  # Elton Group
    "PLAKR.AT",  # Plaisio
    "VIS.AT",    # Vis
    "BIOKA.AT",  # Biokarystis
    # Tourism/Travel
    "AEGN.AT",   # Aegean Airlines
    "ANEK.AT",   # ANEK Lines
    "MINOA.AT",  # Minoan Lines
    "ATTICA.AT", # Attica Group
    # Marine/Shipping
    "STRIK.AT",  # Strintzis Lines
    "TSOUK.AT",  # J. P. Tsoukalis
    # Real Estate/Construction
    "INTRK.AT",  # Intracom Holdings
    "AVAX.AT",   # AVAX
    "DOMIK.AT",  # Domiki Kritis
    "ATTICA.AT", # Attica Publications
    # Others
    "SAR.AT",    # Sarantis
    "MOH.AT",    # Motor Oil Hellas
    "EYAPS.AT",  # Thessaloniki Water
    "EYDAP.AT",  # Athens Water Supply
    "OLTH.AT",   # Thessaloniki Port
    "PPA.AT",    # Piraeus Port
    "ATTICA.AT", # Attica Publications
    "QUEST.AT"   # Quest Holdings
]

# Time range - 5 years
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Approximately 5 years

# Create directory if it doesn't exist
data_dir = 'GREECE/GREECE_STOCK_DATA'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Fetch and save data
successful_downloads = []
failed_downloads = []

for ticker in tqdm(tickers, desc="Downloading Greek stock data"):
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

print("\nAll Greek stock data saved in the 'GREECE/GREECE_STOCK_DATA' folder!") 