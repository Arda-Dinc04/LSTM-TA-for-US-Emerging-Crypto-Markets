import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.dates import YearLocator, DateFormatter

# Set the style
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Read the AMT data
amt_df = pd.read_csv('CRYPTO_STOCKS/AMT_5y_technical_analysis.csv')

# Convert dates to datetime
amt_df['Date'] = pd.to_datetime(amt_df['Date'])

# Sort by date to ensure proper plotting
amt_df = amt_df.sort_values('Date')

# Calculate the train/test split point (80% train, 20% test)
amt_split_idx = int(len(amt_df) * 0.8)

# Create the visualization
plt.figure(figsize=(15, 8))

# Split data into train and test
amt_train = amt_df.iloc[:amt_split_idx]
amt_test = amt_df.iloc[amt_split_idx:]

# Plot AMT with different colors for train and test
plt.plot(amt_train['Date'], amt_train['Close'], label='Training Data', color='blue', alpha=0.7)
plt.plot(amt_test['Date'], amt_test['Close'], label='Testing Data', color='red', alpha=0.7)

# Add vertical line at split point
split_date = amt_df.iloc[amt_split_idx]['Date']
plt.axvline(x=split_date, color='gray', linestyle='--', alpha=0.5)

# Customize the plot
plt.title('AMT Stock Price Movement with Train/Test Split (80/20)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Format x-axis dates
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout and save
plt.tight_layout()
plt.savefig('amt_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Print statistics
print("\nAMT Analysis:")
print("\nTime Periods:")
print(f"Training period: {amt_train['Date'].iloc[0].strftime('%Y-%m-%d')} to {amt_train['Date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Testing period: {amt_test['Date'].iloc[0].strftime('%Y-%m-%d')} to {amt_test['Date'].iloc[-1].strftime('%Y-%m-%d')}")

print("\nPrice Statistics:")
print(f"Training price range: ${amt_train['Close'].min():.2f} - ${amt_train['Close'].max():.2f}")
print(f"Testing price range: ${amt_test['Close'].min():.2f} - ${amt_test['Close'].max():.2f}")

print("\nVolatility Analysis:")
print(f"Training period std dev: ${amt_train['Close'].std():.2f}")
print(f"Testing period std dev: ${amt_test['Close'].std():.2f}")

print("\nPrice Change Analysis:")
print(f"Training period price change: ${amt_train['Close'].iloc[-1] - amt_train['Close'].iloc[0]:.2f}")
print(f"Testing period price change: ${amt_test['Close'].iloc[-1] - amt_test['Close'].iloc[0]:.2f}")

# Calculate and print percentage of days with price increase
amt_train.loc[:, 'Price_Increase'] = amt_train['Close'] > amt_train['Close'].shift(1)
amt_test.loc[:, 'Price_Increase'] = amt_test['Close'] > amt_test['Close'].shift(1)

print("\nUpward Movement Days:")
print(f"Training period: {amt_train['Price_Increase'].mean()*100:.1f}%")
print(f"Testing period: {amt_test['Price_Increase'].mean()*100:.1f}%") 