import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from technical_features import prepare_technical_features

# Configuration
SEQUENCE_LENGTH = 100  # Increased from 60 to 100 for longer historical context
FUTURE_PERIOD = 1    # Number of days to predict ahead
TEST_SIZE = 0.2      # Proportion of data for testing
VALIDATION_SIZE = 0.2  # Proportion of training data for validation
RANDOM_SEED = 42
EPOCHS = 50          # Increased from 25 to 50
BATCH_SIZE = 64      # Increased from 32 to 64

def prepare_data(df, sequence_length):
    """Prepare data for LSTM model with technical features and consistent price scaling"""
    # Add technical features
    df = prepare_technical_features(df)
    
    # Define feature columns (all except Date and target)
    feature_columns = [col for col in df.columns if col != 'Date' and col != 'Close']
    target_column = 'Close'
    
    # Create separate scalers for features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler(feature_range=(-1, 1))  # Consistent range for prices
    
    # Scale features and target separately
    scaled_features = feature_scaler.fit_transform(df[feature_columns])
    scaled_target = target_scaler.fit_transform(df[[target_column]])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length - FUTURE_PERIOD + 1):
        X.append(scaled_features[i:(i + sequence_length)])
        y.append(scaled_target[i + sequence_length + FUTURE_PERIOD - 1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train, validation, and test sets
    train_size = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_size = int(len(X_train) * (1 - VALIDATION_SIZE))
    X_train, X_val = X_train[:train_size], X_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_scaler, target_scaler, feature_columns

def create_lstm_model(input_shape):
    """Create LSTM model with L2 regularization and increased dropout"""
    l2_reg = tf.keras.regularizers.l2(0.01)  # L2 regularization
    
    model = Sequential([
        LSTM(200,  # Increased from 100 to 200 units
             activation='relu', 
             input_shape=input_shape, 
             return_sequences=True,
             kernel_regularizer=l2_reg,
             recurrent_regularizer=l2_reg),
        Dropout(0.3),
        LSTM(200,  # Increased from 100 to 200 units
             activation='relu',
             kernel_regularizer=l2_reg,
             recurrent_regularizer=l2_reg),
        Dropout(0.3),
        Dense(100,  # Increased from 50 to 100 units
             activation='relu',
             kernel_regularizer=l2_reg),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Use a lower learning rate with regularization
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X_test, y_test, target_scaler):
    """Calculate evaluation metrics"""
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = target_scaler.inverse_transform(predictions)
    y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'predictions': predictions.flatten(),
        'actual': y_test.flatten()
    }

def plot_training_history(history, stock_name, output_dir):
    """Plot training history with enhanced visualization"""
    plt.figure(figsize=(15, 7))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title(f'Training History - {stock_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot loss ratio
    plt.subplot(1, 2, 2)
    loss_ratio = np.array(history.history['val_loss']) / np.array(history.history['loss'])
    plt.plot(loss_ratio, label='Validation/Training Loss Ratio', color='green', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Ideal Ratio')
    plt.title('Validation/Training Loss Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{stock_name}_training_history.png'))
    plt.close()

def plot_predictions(predictions, actual, stock_name, output_dir):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'Predictions vs Actual - {stock_name}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{stock_name}_predictions.png'))
    plt.close()

def main():
    # Create output directory for results
    output_dir = "LSTM_Results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a subset of diverse stocks from different sectors
    selected_stocks = [
        'AAPL',  # Technology
        'JPM',   # Financial Services
        'JNJ',   # Healthcare
        'XOM',   # Energy
        'WMT',   # Consumer Goods
        'AMZN',  # E-commerce
        'DIS'    # Entertainment & Media
    ]
    
    # Process each stock in the US Stocks directory
    stock_files = [f for f in os.listdir("US Stocks") if f.split('_')[0] in selected_stocks]
    results = {}
    
    print(f"\nProcessing {len(stock_files)} stocks: {', '.join(selected_stocks)}\n")
    
    for stock_file in stock_files:
        stock_name = stock_file.split('_')[0]
        print(f"\nProcessing {stock_name}...")
        
        # Load stock data
        stock_data = pd.read_csv(os.path.join("US Stocks", stock_file))
        
        # Create stock-specific output directory
        stock_output_dir = os.path.join(output_dir, stock_name)
        os.makedirs(stock_output_dir, exist_ok=True)
        
        try:
            # Prepare data with technical features
            (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_scaler, target_scaler, feature_columns = prepare_data(
                stock_data, SEQUENCE_LENGTH
            )
            
            # Create and train model with early stopping
            model = create_lstm_model(input_shape=(SEQUENCE_LENGTH, X_train.shape[2]))
            
            # Add early stopping with increased patience
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,     # Increased from 5 to 10
                restore_best_weights=True
            )
            
            # Add learning rate reduction on plateau with increased patience
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=6,      # Increased from 3 to 6
                min_lr=0.00001   # Decreased from 0.0001 to 0.00001 for finer convergence
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate model
            eval_results = evaluate_model(model, X_test, y_test, target_scaler)
            
            # Plot results
            plot_training_history(history, stock_name, stock_output_dir)
            plot_predictions(eval_results['predictions'], eval_results['actual'], 
                           stock_name, stock_output_dir)
            
            # Calculate additional metrics
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            overfitting_ratio = final_val_loss / final_train_loss
            
            # Calculate directional accuracy
            actual_direction = np.diff(eval_results['actual']) > 0
            pred_direction = np.diff(eval_results['predictions']) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Store results
            results[stock_name] = {
                'RMSE': eval_results['RMSE'],
                'MAE': eval_results['MAE'],
                'Directional_Accuracy': directional_accuracy,
                'Features': len(feature_columns),
                'Final_Train_Loss': final_train_loss,
                'Final_Val_Loss': final_val_loss,
                'Overfitting_Ratio': overfitting_ratio,
                'Epochs_Trained': len(history.history['loss'])
            }
            
            print(f"Successfully processed {stock_name}")
            print(f"RMSE: {eval_results['RMSE']:.2f}")
            print(f"MAE: {eval_results['MAE']:.2f}")
            print(f"Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"Overfitting Ratio: {overfitting_ratio:.2f}")
            
        except Exception as e:
            print(f"Error processing {stock_name}: {str(e)}")
            results[stock_name] = {
                'RMSE': None,
                'MAE': None,
                'Directional_Accuracy': None,
                'Features': None,
                'Error': str(e)
            }
    
    # Save overall results to CSV
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_dir, 'overall_results.csv'))
    
    # Print final accuracy report
    print("\n" + "="*50)
    print("FINAL ACCURACY REPORT")
    print("="*50)
    print("\nStock Performance Summary:")
    print("-"*50)
    
    for stock_name, metrics in results.items():
        if metrics['RMSE'] is not None:
            print(f"\n{stock_name}:")
            print(f"  RMSE: {metrics['RMSE']:.2f}")
            print(f"  MAE: {metrics['MAE']:.2f}")
            print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
            print(f"  Epochs Trained: {metrics['Epochs_Trained']}")
            print(f"  Overfitting Ratio: {metrics['Overfitting_Ratio']:.2f}")
    
    # Calculate and print average metrics
    successful_stocks = {k: v for k, v in results.items() if v['RMSE'] is not None}
    if successful_stocks:
        avg_rmse = np.mean([v['RMSE'] for v in successful_stocks.values()])
        avg_mae = np.mean([v['MAE'] for v in successful_stocks.values()])
        avg_dir_acc = np.mean([v['Directional_Accuracy'] for v in successful_stocks.values()])
        
        print("\n" + "-"*50)
        print("Average Metrics Across All Stocks:")
        print(f"  Average RMSE: {avg_rmse:.2f}")
        print(f"  Average MAE: {avg_mae:.2f}")
        print(f"  Average Directional Accuracy: {avg_dir_acc:.2f}%")
    
    print("\nTraining and evaluation complete! Results are saved in the LSTM_Results directory.")

if __name__ == "__main__":
    main() 