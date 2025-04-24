import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from technical_features import prepare_technical_features

# Enhanced Configuration
SEQUENCE_LENGTH = 60
FUTURE_PERIOD = 1
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LSTM_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.3

def create_attention_mask(inputs):
    """Create attention mask for self-attention mechanism"""
    mask = tf.cast(tf.math.not_equal(inputs, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def prepare_data(df, sequence_length):
    """Enhanced data preparation with improved scaling"""
    df = prepare_technical_features(df)
    
    feature_columns = [col for col in df.columns if col != 'Date' and col != 'Close']
    target_column = 'Close'
    
    # Separate scalers for different types of features
    price_scaler = MinMaxScaler(feature_range=(-1, 1))
    volume_scaler = MinMaxScaler(feature_range=(-1, 1))
    indicator_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Scale different types of features separately
    price_features = ['Open', 'High', 'Low', 'Close']
    volume_features = ['Volume']
    indicator_features = [col for col in feature_columns if col not in price_features + volume_features]
    
    scaled_prices = price_scaler.fit_transform(df[price_features])
    scaled_volume = volume_scaler.fit_transform(df[volume_features])
    scaled_indicators = indicator_scaler.fit_transform(df[indicator_features])
    
    # Combine scaled features
    scaled_features = np.hstack((scaled_prices, scaled_volume, scaled_indicators))
    
    # Create sequences with improved handling
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length - FUTURE_PERIOD + 1):
        X.append(scaled_features[i:(i + sequence_length)])
        y.append(scaled_prices[i + sequence_length + FUTURE_PERIOD - 1, price_features.index('Close')])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    train_size = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_size = int(len(X_train) * (1 - VALIDATION_SIZE))
    X_train, X_val = X_train[:train_size], X_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), price_scaler, feature_columns

def create_advanced_model(input_shape):
    """Create optimized LSTM model without bidirectional layers"""
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(LSTM_UNITS, 
             return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Simplified attention mechanism
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=LSTM_UNITS//2
    )(x, x, x)
    x = tf.keras.layers.Add()([x, attention_output])
    
    # Second LSTM layer
    x = LSTM(LSTM_UNITS // 2)(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Dense layer
    x = Dense(DENSE_UNITS, activation='relu')(x)
    x = Dropout(DROPOUT_RATE/2)(x)
    
    outputs = Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X_test, y_test, price_scaler):
    """Enhanced model evaluation with additional metrics"""
    predictions = model.predict(X_test)
    
    # Reshape for inverse transform
    y_test_reshaped = y_test.reshape(-1, 1)
    predictions_reshaped = predictions.reshape(-1, 1)
    
    # Inverse transform predictions and actual values
    y_test_orig = price_scaler.inverse_transform(np.hstack([np.zeros((len(y_test), 3)), y_test_reshaped]))[:, 3]
    pred_orig = price_scaler.inverse_transform(np.hstack([np.zeros((len(predictions), 3)), predictions_reshaped]))[:, 3]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
    mae = mean_absolute_error(y_test_orig, pred_orig)
    mape = np.mean(np.abs((y_test_orig - pred_orig) / y_test_orig)) * 100
    
    # Calculate directional accuracy
    actual_direction = np.diff(y_test_orig) > 0
    pred_direction = np.diff(pred_orig) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy,
        'predictions': pred_orig,
        'actual': y_test_orig
    }

def plot_results(history, predictions, actual, stock_name, output_dir):
    """Enhanced visualization of results"""
    # Create stock-specific output directory
    stock_dir = os.path.join(output_dir, stock_name)
    os.makedirs(stock_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{stock_name} - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.plot(actual, label='Actual', alpha=0.8)
    plt.plot(predictions, label='Predicted', alpha=0.8)
    plt.title(f'{stock_name} - Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(stock_dir, f'{stock_name}_results.png'))
    plt.close()

def main():
    """Main function to run the enhanced LSTM model"""
    output_dir = "LSTM_v2_Results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Selected stocks from different sectors
    selected_stocks = [
        'AAPL'  # Technology
    ]
    
    results = {}
    print(f"\nProcessing {len(selected_stocks)} stocks with enhanced LSTM model v2")
    
    for stock_name in selected_stocks:
        print(f"\nProcessing {stock_name}...")
        try:
            # Load and prepare data
            stock_data = pd.read_csv(os.path.join("US Stocks", f"{stock_name}_5y_technical_analysis.csv"))
            (X_train, y_train), (X_val, y_val), (X_test, y_test), price_scaler, feature_columns = prepare_data(
                stock_data, SEQUENCE_LENGTH
            )
            
            # Create and train model
            model = create_advanced_model(input_shape=(SEQUENCE_LENGTH, X_train.shape[2]))
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.00001
                )
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate and plot results
            eval_results = evaluate_model(model, X_test, y_test, price_scaler)
            plot_results(history, eval_results['predictions'], eval_results['actual'], 
                        stock_name, output_dir)
            
            # Store results
            results[stock_name] = {
                'RMSE': eval_results['RMSE'],
                'MAE': eval_results['MAE'],
                'MAPE': eval_results['MAPE'],
                'Directional_Accuracy': eval_results['Directional_Accuracy'],
                'Features': len(feature_columns),
                'Final_Train_Loss': history.history['loss'][-1],
                'Final_Val_Loss': history.history['val_loss'][-1]
            }
            
            print(f"Successfully processed {stock_name}")
            print(f"RMSE: {eval_results['RMSE']:.2f}")
            print(f"MAE: {eval_results['MAE']:.2f}")
            print(f"MAPE: {eval_results['MAPE']:.2f}%")
            print(f"Directional Accuracy: {eval_results['Directional_Accuracy']:.2f}%")
            
        except Exception as e:
            print(f"Error processing {stock_name}: {str(e)}")
            results[stock_name] = {'Error': str(e)}
    
    # Save and display final results
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_dir, 'v2_overall_results.csv'))
    
    print("\n" + "="*50)
    print("FINAL ACCURACY REPORT - LSTM v2")
    print("="*50)
    
    successful_stocks = {k: v for k, v in results.items() if 'Error' not in v}
    if successful_stocks:
        avg_rmse = np.mean([v['RMSE'] for v in successful_stocks.values()])
        avg_mae = np.mean([v['MAE'] for v in successful_stocks.values()])
        avg_mape = np.mean([v['MAPE'] for v in successful_stocks.values()])
        avg_dir_acc = np.mean([v['Directional_Accuracy'] for v in successful_stocks.values()])
        
        print("\nAverage Metrics Across All Stocks:")
        print(f"Average RMSE: {avg_rmse:.2f}")
        print(f"Average MAE: {avg_mae:.2f}")
        print(f"Average MAPE: {avg_mape:.2f}%")
        print(f"Average Directional Accuracy: {avg_dir_acc:.2f}%")
    
    print("\nTraining and evaluation complete! Results are saved in the LSTM_v2_Results directory.")

if __name__ == "__main__":
    main()
