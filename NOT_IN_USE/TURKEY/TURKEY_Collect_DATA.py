import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# List of top Turkish stocks by sector
tickers = [
    # Banking/Finance
    "AKBNK.IS",  # Akbank
    "GARAN.IS",  # Garanti BBVA
    "ISCTR.IS",  # İş Bankası
    "YKBNK.IS",  # Yapı Kredi
    "VAKBN.IS",  # Vakıfbank
    "HALKB.IS",  # Halkbank
    "THYAO.IS",  # Turkish Airlines
    # Energy/Utilities
    "TUPRS.IS",  # Tüpraş
    "AKSEN.IS",  # Aksa Enerji
    "ENJSA.IS",  # Enerjisa
    "AKENR.IS",  # Ak Enerji
    "AYGAZ.IS",  # Aygaz
    # Telecom/Technology
    "TCELL.IS",  # Turkcell
    "TTKOM.IS",  # Türk Telekom
    "NETAS.IS",  # Netaş
    "LOGO.IS",   # Logo Yazılım
    "ARENA.IS",  # Arena Bilgisayar
    # Retail/Consumer
    "BIMAS.IS",  # BİM
    "MGROS.IS",  # Migros
    "SOKM.IS",   # Şok Marketler
    "DOAS.IS",   # Doğuş Otomotiv
    "MAVI.IS",   # Mavi
    # Industrial/Manufacturing
    "ARCLK.IS",  # Arçelik
    "TOASO.IS",  # Tofaş
    "VESTL.IS",  # Vestel
    "FROTO.IS",  # Ford Otosan
    "OTKAR.IS",  # Otokar
    "KORDS.IS",  # Kordsa
    # Construction/Real Estate
    "TKFEN.IS",  # Tekfen Holding
    "EREGL.IS",  # Ereğli Demir Çelik
    "SISE.IS",   # Şişecam
    "KCHOL.IS",  # Koç Holding
    "SAHOL.IS",  # Sabancı Holding
    "ECILC.IS",  # EİS Eczacıbaşı
    # Healthcare/Pharma
    "DEVA.IS",   # DEVA Holding
    "SELEC.IS",  # Selçuk Ecza
    "MPARK.IS",  # MLP Care
    # Food/Beverage
    "ULKER.IS",  # Ülker
    "AEFES.IS",  # Anadolu Efes
    "PNSUT.IS",  # Pınar Süt
    "TATGD.IS",  # Tat Gıda
    "CCOLA.IS",  # Coca-Cola İçecek
    # Others
    "TAVHL.IS",  # TAV Havalimanları
    "PGSUS.IS",  # Pegasus
    "ASELS.IS",  # Aselsan
    "GUBRF.IS",  # Gübre Fabrikaları
    "PETKM.IS",  # Petkim
    "ALBRK.IS",  # Albaraka Türk
    "KARSN.IS",  # Karsan
    "TSKB.IS"    # Türkiye Sınai Kalkınma Bankası
]

# Time range - 5 years
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Approximately 5 years

# Create directory if it doesn't exist
data_dir = 'TURKEY/TURKEY_STOCK_DATA'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Fetch and save data
successful_downloads = []
failed_downloads = []

for ticker in tqdm(tickers, desc="Downloading Turkish stock data"):
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

print("\nAll Turkish stock data saved in the 'TURKEY/TURKEY_STOCK_DATA' folder!") 