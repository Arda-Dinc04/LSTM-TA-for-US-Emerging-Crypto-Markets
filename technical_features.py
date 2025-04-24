import pandas as pd
import numpy as np

def calculate_sma_features(df):
    """Calculate SMA-based features"""
    # Golden/Death Cross and SMA trends
    df['SMA_Cross'] = 0
    df.loc[(df['SMA50'] > df['SMA200']) & (df['SMA50'].shift(1) <= df['SMA200'].shift(1)), 'SMA_Cross'] = 1  # Golden
    df.loc[(df['SMA50'] < df['SMA200']) & (df['SMA50'].shift(1) >= df['SMA200'].shift(1)), 'SMA_Cross'] = -1 # Death

    # SMA Trends (slope over 5 days)
    df['SMA50_Trend'] = df['SMA50'].diff(5) / df['SMA50'].shift(5)
    df['SMA200_Trend'] = df['SMA200'].diff(5) / df['SMA200'].shift(5)
    
    # Normalized distance between SMAs
    df['SMA_Distance'] = (df['SMA50'] - df['SMA200']) / df['Close']
    
    return df

def calculate_rsi_features(df):
    """Calculate RSI-based features"""
    # Overbought/Oversold signals
    df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
    
    # RSI crosses
    df['RSI_Cross_Up'] = ((df['RSI'] > 30) & (df['RSI'].shift(1) < 30)).astype(int)
    df['RSI_Cross_Down'] = ((df['RSI'] < 70) & (df['RSI'].shift(1) > 70)).astype(int)
    
    # RSI Divergence (simplified)
    price_highs = df['Close'].rolling(window=14).max()
    rsi_highs = df['RSI'].rolling(window=14).max()
    df['RSI_Divergence'] = ((price_highs > price_highs.shift(1)) & 
                           (rsi_highs < rsi_highs.shift(1))).astype(int)
    
    return df

def calculate_macd_features(df):
    """Calculate MACD-based features"""
    # Crossovers
    df['MACD_Bullish_Cross'] = ((df['MACD'] > df['MACD_Signal']) & 
                               (df['MACD'].shift(1) < df['MACD_Signal'].shift(1))).astype(int)
    df['MACD_Bearish_Cross'] = ((df['MACD'] < df['MACD_Signal']) & 
                               (df['MACD'].shift(1) > df['MACD_Signal'].shift(1))).astype(int)
    
    # Zero line crosses
    df['MACD_Above_Zero'] = (df['MACD'] > 0).astype(int)
    df['MACD_Below_Zero'] = (df['MACD'] < 0).astype(int)
    
    # Histogram momentum
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Histogram_Change'] = df['MACD_Histogram'].diff()
    
    return df

def calculate_adx_features(df):
    """Calculate ADX-based features"""
    # Trend strength
    df['Strong_Trend'] = (df['ADX'] > 25).astype(int)
    df['Weak_Trend'] = (df['ADX'] < 20).astype(int)
    
    # DI crosses (if available)
    if 'ADX_Plus_DI' in df.columns and 'ADX_Minus_DI' in df.columns:
        df['DI_Bullish'] = (df['ADX_Plus_DI'] > df['ADX_Minus_DI']).astype(int)
        df['DI_Bearish'] = (df['ADX_Plus_DI'] < df['ADX_Minus_DI']).astype(int)
        df['DI_Cross'] = ((df['ADX_Plus_DI'] > df['ADX_Minus_DI']) & 
                         (df['ADX_Plus_DI'].shift(1) < df['ADX_Minus_DI'].shift(1))).astype(int)
    
    return df


def calculate_bollinger_features(df):
    """Calculate Bollinger Bands-based features"""
    # Overbought/Oversold signals
    df['BB_Overbought'] = (df['Close'] > df['BB_Upper']).astype(int)
    df['BB_Oversold'] = (df['Close'] < df['BB_Lower']).astype(int)
    
    # Band width and squeeze (low volatility)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(window=20).mean()).astype(int)
    
    # Percentile rank of width — helps detect relative narrow bands
    df['BB_Width_PctRank'] = df['BB_Width'].rank(pct=True)

    # Mean reversion signal
    df['BB_Mean_Reversion'] = ((df['Close'] - df['BB_Middle']).abs() / 
                              (df['BB_Upper'] - df['BB_Middle']))
    
    return df


def prepare_technical_features(df):
    """Prepare all technical features for the LSTM model"""
    df = calculate_sma_features(df)
    df = calculate_rsi_features(df)
    df = calculate_macd_features(df)
    df = calculate_adx_features(df)
    df = calculate_bollinger_features(df)
    
    # Drop any NaN values created by calculations
    df = df.dropna()
    
    return df 