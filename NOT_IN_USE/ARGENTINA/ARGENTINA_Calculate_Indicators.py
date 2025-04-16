import pandas as pd
import pandas_ta as ta
import os
from tqdm import tqdm
import numpy as np

def add_indicators_safely(df):
    """Add all technical indicators with proper error handling for each indicator."""
    try:
        # Calculate SMAs
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Calculate MACD
        macd = ta.macd(df['Close'])
        if not macd.empty:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_Signal'] = macd['MACDs_12_26_9']
            df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # Calculate Bollinger Bands
        bb = ta.bbands(df['Close'])
        if not bb.empty:
            df['BB_Upper'] = bb['BBU_5_2.0']  # Upper band
            df['BB_Middle'] = bb['BBM_5_2.0']  # Middle band
            df['BB_Lower'] = bb['BBL_5_2.0']  # Lower band
        
        # Calculate ADX
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        if not adx.empty:
            df['ADX'] = adx['ADX_14']
            df['ADX_Plus_DI'] = adx['DMP_14']
            df['ADX_Minus_DI'] = adx['DMN_14']
        
        # Calculate EMAs
        df['EMA_9'] = ta.ema(df['Close'], length=9)
        df['EMA_21'] = ta.ema(df['Close'], length=21)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['EMA_200'] = ta.ema(df['Close'], length=200)
        
    except Exception as e:
        print(f"Warning: Error during indicator calculation: {e}")
    
    return df

def process_stock_files():
    # Get list of all CSV files in BRAZIL_STOCK_DATA directory
    data_dir = 'ARGENTINA/ARGENTINA_STOCK_DATA'
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('temp_')]
    
    processed_count = 0
    failed_count = 0
    
    for file in tqdm(files, desc="Processing argentina stocks"):
        try:
            # Read the file content
            with open(f'{data_dir}/{file}', 'r') as f:
                content = f.readlines()
            
            # Create a new header structure
            price_header = content[0].strip().split(',')
            ticker_header = content[1].strip().split(',')
            date_header = content[2].strip().split(',')
            
            # Extend headers with indicator names
            indicator_names = [
                'SMA_50', 'SMA_200', 'RSI', 
                'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Upper', 'BB_Middle', 'BB_Lower',
                'ADX', 'ADX_Plus_DI', 'ADX_Minus_DI',
                'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200'
            ]
            
            # Extend the price header row
            while len(price_header) < 6 + len(indicator_names):
                price_header.append('Indicator')
            
            # Extend the ticker header row
            while len(ticker_header) < 6 + len(indicator_names):
                ticker_header.append(ticker_header[1])  # Use the same ticker
                
            # Extend the date header row
            while len(date_header) < 6 + len(indicator_names):
                date_header.append('')
                
            # Update the indicator names in the first row
            for i, name in enumerate(indicator_names):
                price_header[6 + i] = name
            
            # Create a temporary CSV file without headers
            temp_file = f'{data_dir}/temp_{file}'
            with open(temp_file, 'w') as f:
                f.writelines(content[3:])
            
            # Read the temp file with correct column names
            df = pd.read_csv(temp_file, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Set Date as index for TA calculations
            df.set_index('Date', inplace=True)
            
            # Convert price columns to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Add technical indicators safely
            df = add_indicators_safely(df)
            
            # Reset index to get Date back as column
            df.reset_index(inplace=True)
            
            # Create updated CSV with new headers and data with indicators
            with open(f'{data_dir}/{file}', 'w') as f:
                # Write updated header rows
                f.write(','.join(price_header) + '\n')
                f.write(','.join(ticker_header) + '\n')
                f.write(','.join(date_header) + '\n')
                
                # Write data rows with indicators
                df.to_csv(f, index=False, header=False)
            
            # Remove temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            processed_count += 1
            print(f"✓ Processed {file}")
            
        except Exception as e:
            failed_count += 1
            print(f"✗ Error processing {file}: {str(e)}")
    
    print(f"\nSuccessfully processed: {processed_count} files")
    print(f"Failed to process: {failed_count} files")

if __name__ == "__main__":
    process_stock_files()
    print("\nAll argentina stocks processed!") 