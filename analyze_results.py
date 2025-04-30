import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Stock sector mapping
STOCKS = {
    # Technology
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'META': 'Technology',
    'NVDA': 'Technology',
    'AVGO': 'Technology',
    'ORCL': 'Technology',
    'CRM': 'Technology',
    'AMD': 'Technology',
    'INTC': 'Technology',
    
    # Healthcare
    'JNJ': 'Healthcare',
    'UNH': 'Healthcare',
    'LLY': 'Healthcare',
    'PFE': 'Healthcare',
    'ABT': 'Healthcare',
    
    # Financial Services
    'JPM': 'Financial Services',
    'BAC': 'Financial Services',
    'V': 'Financial Services',
    'MA': 'Financial Services',
    'WFC': 'Financial Services',
    
    # Consumer Goods
    'PG': 'Consumer Goods',
    'KO': 'Consumer Goods',
    'PEP': 'Consumer Goods',
    'COST': 'Consumer Goods',
    'WMT': 'Consumer Goods',
    
    # Industrial
    'CAT': 'Industrial',
    'BA': 'Industrial',
    'HON': 'Industrial',
    'UPS': 'Industrial',
    'GE': 'Industrial',
    
    # Energy
    'XOM': 'Energy',
    'CVX': 'Energy',
    'COP': 'Energy',
    'SLB': 'Energy',
    'EOG': 'Energy',
    
    # Telecommunications
    'T': 'Telecommunications',
    'VZ': 'Telecommunications',
    'TMUS': 'Telecommunications',
    
    # Real Estate
    'AMT': 'Real Estate',
    'PLD': 'Real Estate',
    'CCI': 'Real Estate',
    
    # Materials
    'LIN': 'Materials',
    'APD': 'Materials',
    'FCX': 'Materials',
    
    # Utilities
    'NEE': 'Utilities',
    'DUK': 'Utilities',
    'SO': 'Utilities',
    
    # Entertainment & Media
    'NFLX': 'Entertainment & Media',
    'DIS': 'Entertainment & Media',
    'CMCSA': 'Entertainment & Media',
    
    # E-commerce
    'AMZN': 'E-commerce',
    'EBAY': 'E-commerce',
    'ETSY': 'E-commerce'
}

def parse_results_file(file_path):
    """Parse the results file and extract performance metrics."""
    data = []
    current_asset = None
    metrics = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if ':' in line and not any(x in line for x in ['•', 'Ranked by']):
                if current_asset and metrics:
                    data.append({'Asset': current_asset, **metrics})
                current_asset = line.replace(':', '').strip()
                metrics = {}
            elif '•' in line:
                metric, value = line.split('•')[1].split(':')
                metric = metric.strip()
                value = value.strip()
                if '%' in value:
                    value = float(value.replace('%', ''))
                else:
                    value = float(value)
                metrics[metric] = value
    
    # Add the last asset
    if current_asset and metrics:
        data.append({'Asset': current_asset, **metrics})
    
    df = pd.DataFrame(data)
    
    # Add asset type
    df['Asset Type'] = df['Asset'].apply(lambda x: 'Crypto' if 'USD' in x else 'Stock')
    
    # Add sector for stocks
    def get_sector(asset):
        if 'USD' in asset:  # For crypto assets
            return 'Cryptocurrency'
        # For stocks, extract ticker from the technical analysis suffix
        ticker = asset.split('_')[0]
        return STOCKS.get(ticker, 'Other')
    
    df['Sector'] = df['Asset'].apply(get_sector)
    
    return df

def create_visualizations(df):
    """Create various visualizations from the performance data."""
    # Set style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Create output directory if it doesn't exist
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Calculate outlier thresholds using IQR method
    def get_outlier_thresholds(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
    rmse_lower, rmse_upper = get_outlier_thresholds(df, 'RMSE')
    mape_lower, mape_upper = get_outlier_thresholds(df, 'MAPE')
    
    # Create filtered dataframe without outliers
    df_filtered = df[
        (df['RMSE'] >= rmse_lower) & 
        (df['RMSE'] <= rmse_upper) & 
        (df['MAPE'] >= mape_lower) & 
        (df['MAPE'] <= mape_upper)
    ]
    
    # Print outlier information
    print("\nOutlier Analysis:")
    print(f"RMSE range (without outliers): {rmse_lower:.2f} to {rmse_upper:.2f}")
    print(f"MAPE range (without outliers): {mape_lower:.2f} to {mape_upper:.2f}")
    print("\nOutliers removed:")
    outliers = df[~df.index.isin(df_filtered.index)]
    print(outliers[['Asset', 'RMSE', 'MAPE', 'Sector']].to_string())
    
    # 1. RMSE vs MAPE by Sector - Full Range
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=df, x='RMSE', y='MAPE', hue='Sector', size='Composite Score',
                   sizes=(50, 400), alpha=0.7)
    plt.title('RMSE vs MAPE by Sector - Full Range (Size indicates Composite Score)')
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_vs_mape_by_sector_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE vs MAPE by Sector - Without Outliers
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=df_filtered, x='RMSE', y='MAPE', hue='Sector', size='Composite Score',
                   sizes=(50, 400), alpha=0.7)
    plt.title('RMSE vs MAPE by Sector - Without Outliers (Size indicates Composite Score)')
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_vs_mape_by_sector_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. RMSE vs MAPE by Asset Type - Full Range
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=df, x='RMSE', y='MAPE', hue='Asset Type', style='Asset Type',
                   size='Composite Score', sizes=(50, 400), alpha=0.7)
    plt.title('RMSE vs MAPE by Asset Type - Full Range (Size indicates Composite Score)')
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_vs_mape_by_type_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. RMSE vs MAPE by Asset Type - Without Outliers
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=df_filtered, x='RMSE', y='MAPE', hue='Asset Type', style='Asset Type',
                   size='Composite Score', sizes=(50, 400), alpha=0.7)
    plt.title('RMSE vs MAPE by Asset Type - Without Outliers (Size indicates Composite Score)')
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_vs_mape_by_type_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics about the filtered dataset
    stats_filtered = pd.DataFrame({
        'Metric': ['RMSE', 'MAPE'],
        'Mean': [df_filtered['RMSE'].mean(), df_filtered['MAPE'].mean()],
        'Median': [df_filtered['RMSE'].median(), df_filtered['MAPE'].median()],
        'Std': [df_filtered['RMSE'].std(), df_filtered['MAPE'].std()],
        'Min': [df_filtered['RMSE'].min(), df_filtered['MAPE'].min()],
        'Max': [df_filtered['RMSE'].max(), df_filtered['MAPE'].max()]
    })
    stats_filtered.to_csv(output_dir / 'error_metrics_stats_filtered.csv', index=False)
    
    # 5. Top 20 assets by composite score
    plt.figure(figsize=(15, 8))
    top_20 = df.nlargest(20, 'Composite Score')
    sns.barplot(x='Composite Score', y='Asset', data=top_20, hue='Sector', dodge=False)
    plt.title('Top 20 Assets by Composite Score')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_20_assets.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Performance by Sector
    plt.figure(figsize=(12, 6))
    sector_performance = df.groupby('Sector').agg({
        'Composite Score': 'mean',
        'Directional Accuracy': 'mean',
        'RMSE': 'mean',
        'MAPE': 'mean'
    }).round(2)
    
    # Sort sectors by composite score
    sector_performance = sector_performance.sort_values('Composite Score', ascending=False)
    
    # Plot sector performance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sector_performance.index, y='Composite Score', data=sector_performance)
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Composite Score by Sector')
    plt.tight_layout()
    plt.savefig(output_dir / 'sector_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Directional Accuracy Distribution by Asset Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Asset Type', y='Directional Accuracy', data=df)
    plt.title('Directional Accuracy Distribution by Asset Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_asset_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Performance Metrics Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Performance Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Top and Bottom Performers by Sector
    plt.figure(figsize=(15, 10))
    top_bottom = pd.concat([
        df.nlargest(5, 'Composite Score'),
        df.nsmallest(5, 'Composite Score')
    ])
    sns.barplot(x='Composite Score', y='Asset', data=top_bottom, hue='Sector', dodge=False)
    plt.title('Top and Bottom 5 Performers by Composite Score')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_bottom_performers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Save detailed sector analysis to CSV
    sector_analysis = df.groupby('Sector').agg({
        'Composite Score': ['mean', 'std', 'count'],
        'Directional Accuracy': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(4)
    sector_analysis.to_csv(output_dir / 'detailed_sector_analysis.csv')
    
    # 11. Create a performance summary table
    performance_summary = pd.DataFrame({
        'Metric': ['Composite Score', 'Directional Accuracy', 'RMSE', 'MAPE'],
        'Best Performer': [
            df.loc[df['Composite Score'].idxmax(), 'Asset'],
            df.loc[df['Directional Accuracy'].idxmax(), 'Asset'],
            df.loc[df['RMSE'].idxmin(), 'Asset'],
            df.loc[df['MAPE'].idxmin(), 'Asset']
        ],
        'Best Value': [
            df['Composite Score'].max(),
            df['Directional Accuracy'].max(),
            df['RMSE'].min(),
            df['MAPE'].min()
        ],
        'Worst Performer': [
            df.loc[df['Composite Score'].idxmin(), 'Asset'],
            df.loc[df['Directional Accuracy'].idxmin(), 'Asset'],
            df.loc[df['RMSE'].idxmax(), 'Asset'],
            df.loc[df['MAPE'].idxmax(), 'Asset']
        ],
        'Worst Value': [
            df['Composite Score'].min(),
            df['Directional Accuracy'].min(),
            df['RMSE'].max(),
            df['MAPE'].max()
        ]
    })
    performance_summary.to_csv(output_dir / 'performance_summary.csv', index=False)

def main():
    # Parse the results file
    df = parse_results_file('Results')
    
    # Save to CSV
    df.to_csv('model_performance_results.csv', index=False)
    
    # Create visualizations
    create_visualizations(df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nTop 5 Performing Assets:")
    print(df.nlargest(5, 'Composite Score')[['Asset', 'Composite Score', 'Directional Accuracy']])
    
    print("\nBottom 5 Performing Assets:")
    print(df.nsmallest(5, 'Composite Score')[['Asset', 'Composite Score', 'Directional Accuracy']])
    
    print("\nOverall Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main() 