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
SEQUENCE_LENGTH = 100
FUTURE_PERIOD = 1
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 64

def prepare_data(df, sequence_length):
    df = prepare_technical_features(df)

    # Use log return as target
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(FUTURE_PERIOD)).shift(-FUTURE_PERIOD)

    # Add lagged log returns as features
    for lag in range(1, 6):
        df[f'Log_Return_Lag_{lag}'] = np.log(df['Close'] / df['Close'].shift(lag))

    df = df.dropna()

    feature_columns = [col for col in df.columns if col not in ['Date', 'Close', 'Log_Return']]
    target_column = 'Log_Return'

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    close_scaler = MinMaxScaler(feature_range=(0, 1))  # For inverse transforming actual Close prices if needed

    scaled_features = feature_scaler.fit_transform(df[feature_columns])
    scaled_target = target_scaler.fit_transform(df[[target_column]])
    close_scaler.fit(df[['Close']])

    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i + sequence_length])
        y.append(scaled_target[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_size = int(len(X_train) * (1 - VALIDATION_SIZE))
    X_train, X_val = X_train[:train_size], X_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_scaler, target_scaler, feature_columns, close_scaler

def create_lstm_model(input_shape):
    l2_reg = tf.keras.regularizers.l2(0.001)
    model = Sequential([
        LSTM(128, activation='tanh', input_shape=input_shape, return_sequences=True, kernel_regularizer=l2_reg),
        Dropout(0.2),
        LSTM(128, activation='tanh', kernel_regularizer=l2_reg),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X_test, y_test, target_scaler):
    predictions = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)
    y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'predictions': predictions.flatten(),
        'actual': y_test.flatten()
    }

def plot_predictions(predictions, actual, stock_name, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Log Return')
    plt.plot(predictions, label='Predicted Log Return')
    plt.title(f'Log Return Predictions vs Actual - {stock_name}')
    plt.xlabel('Time')
    plt.ylabel('Log Return')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{stock_name}_predicted_log_returns.png'))
    plt.close()

def main():
    output_dir = "LSTM_Return_Results"
    os.makedirs(output_dir, exist_ok=True)

    selected_stocks = ['AAPL']
    stock_files = [f for f in os.listdir("US Stocks") if f.split('_')[0] in selected_stocks]
    results = {}

    for stock_file in stock_files:
        stock_name = stock_file.split('_')[0]
        stock_data = pd.read_csv(os.path.join("US Stocks", stock_file))
        stock_output_dir = os.path.join(output_dir, stock_name)
        os.makedirs(stock_output_dir, exist_ok=True)

        try:
            (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_scaler, target_scaler, feature_columns, close_scaler = prepare_data(
                stock_data, SEQUENCE_LENGTH)

            model = create_lstm_model(input_shape=(SEQUENCE_LENGTH, X_train.shape[2]))

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
            ]

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=1
            )

            eval_results = evaluate_model(model, X_test, y_test, target_scaler)
            plot_predictions(eval_results['predictions'], eval_results['actual'], stock_name, stock_output_dir)

            actual_direction = np.diff(eval_results['actual']) > 0
            pred_direction = np.diff(eval_results['predictions']) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100

            results[stock_name] = {
                'RMSE': eval_results['RMSE'],
                'MAE': eval_results['MAE'],
                'Directional_Accuracy': directional_accuracy,
                'Epochs_Trained': len(history.history['loss'])
            }

        except Exception as e:
            results[stock_name] = {'Error': str(e)}

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_dir, 'return_model_results.csv'))
    print("\nTraining and evaluation complete. Results saved.")

if __name__ == "__main__":
    main()
