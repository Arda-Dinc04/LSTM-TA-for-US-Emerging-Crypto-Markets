# LSTM Trading Assistant for US/Emerging/Crypto Markets

This repository contains financial market data and analysis tools for US stocks, emerging markets, and cryptocurrency markets. The data is processed and prepared for use with LSTM (Long Short-Term Memory) models for market analysis and trading assistance.

## Directory Structure

- `USA/`
  - `USA_STOCK_DATA/` - Contains processed CSV files for US stock market data
    - Each file contains market data including price, volume, and technical indicators
    - Data includes: Open, High, Low, Close, Volume, SMA, RSI, MACD, Bollinger Bands, ADX

## Data Format

Each CSV file in the stock data directories follows this structure:

```
Price,Open,High,Low,Close,Volume,SMA_50,SMA_200,RSI,MACD,MACD_Signal,MACD_Hist,BB_Upper,BB_Middle,BB_Lower,ADX,ADX_Plus_DI,ADX_Minus_DI,EMA_9,EMA_21,EMA_50,EMA_200
```

## Recent Changes

- Removed redundant ticker rows from all stock data files
- Cleaned and standardized CSV file formats
