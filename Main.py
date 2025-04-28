import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ───────── PARAMETERS ─────────
SEQUENCE_LENGTH = 50
EPOCHS = 80
BATCH_SIZE = 32
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

DATA_DIR = 'CRYPTO_STOCKS'

def create_sequences(data, raw_close, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len - 1):
        X.append(data[i:i + seq_len])
        # Calculate returns on raw close prices
        prev_close = raw_close[i + seq_len - 1]
        curr_close = raw_close[i + seq_len]
        if prev_close == 0:
            ret = 0  # Handle division by zero
        else:
            ret = (curr_close - prev_close) / prev_close
        y.append(ret)
    return np.array(X), np.array(y)

# ───────── LOAD & CLEAN ─────────
files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv'))
symbols = [os.path.splitext(f)[0] for f in files]

# Lists to store train/test data per symbol
X_train_list, X_test_list = [], []
y_train_list, y_test_list = [], []
raw_data, scalers = {}, {}

for fname, sym in zip(files, symbols):
    print(f"\nProcessing {sym}...")
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)

    # Handle date column
    if 'date' in df.columns and 'Date' not in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    if 'Date' not in df.columns:
        raise KeyError(f"{fname} has no 'date' or 'Date' column")

    # Keep only Date + features
    cols = ['Date'] + FEATURE_COLS
    df = df.loc[:, [col for col in cols if col in df.columns]]

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', utc=True).dt.tz_convert(None)
    df.sort_values('Date', inplace=True)
    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)

    raw_data[sym] = df.copy()

    # Scale features using RobustScaler
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(df[FEATURE_COLS].values)
    scalers[sym] = scaler

    # Get raw close prices for return calculation
    raw_close = df['Close'].values

    # Build sequences
    X_seq, y_seq = create_sequences(scaled_features, raw_close, SEQUENCE_LENGTH)
    
    # Remove any sequences with NaN or Inf values
    mask = ~np.any(np.isnan(X_seq)) & ~np.any(np.isinf(X_seq)) & ~np.isnan(y_seq) & ~np.isinf(y_seq)
    if np.sum(mask) > 0:
        X_seq = X_seq[mask]
        y_seq = y_seq[mask]
        
        # Split per symbol
        n_train = int(len(X_seq) * 0.8)
        X_train_list.append(X_seq[:n_train])
        X_test_list.append(X_seq[n_train:])
        y_train_list.append(y_seq[:n_train])
        y_test_list.append(y_seq[n_train:])
    else:
        print(f"Warning: {sym} has no valid sequences after cleaning")

if not X_train_list:
    raise ValueError("No valid data found in any of the input files")

# Concatenate all train/test data
X_train = np.concatenate(X_train_list)
X_test = np.concatenate(X_test_list)
y_train = np.concatenate(y_train_list)
y_test = np.concatenate(y_test_list)

# Clip extreme returns
y_train = np.clip(y_train, -1, 1)
y_test = np.clip(y_test, -1, 1)

# ───────── ENSEMBLE TRAINING ─────────
predictions = []
dropouts = [0.2, 0.25, 0.15]
seeds = [42, 7, 99]

for d, s in zip(dropouts, seeds):
    print(f"\nTraining model with dropout {d}, seed {s}")
    np.random.seed(s)
    tf.random.set_seed(s)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURE_COLS))),
        Dropout(d),
        LSTM(50),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='huber')
    
    cbs = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )
    
    predictions.append(model.predict(X_test))

# Average the 3 models
y_pred_avg = np.mean(predictions, axis=0).flatten()

# ───────── EVALUATE & PLOT ─────────
# Track where each symbol's test data starts and ends
test_starts = np.cumsum([0] + [len(y) for y in y_test_list])
test_ranges = list(zip(test_starts[:-1], test_starts[1:]))

# Store metrics for ranking
metrics_data = []

for sym, (start, end) in zip(symbols, test_ranges):
    if sym not in raw_data:
        continue
        
    df = raw_data[sym]
    preds = y_pred_avg[start:end]
    actual = y_test[start:end]
    
    # Mask out any NaN/Inf
    mask = (~np.isnan(preds) & ~np.isnan(actual) & ~np.isinf(preds) & ~np.isinf(actual))
    preds, actual = preds[mask], actual[mask]
    
    if len(preds) == 0:
        print(f"⚠️ {sym}: no valid test points")
        continue
    
    # Directional accuracy
    dir_acc = np.mean((preds > 0) == (actual > 0)) * 100
    
    # Rebuild prices from returns
    closes = df['Close'].values
    n_test = len(preds)
    window = closes[-n_test-1:-1]
    y_true_p = window * (1 + actual)
    y_pred_p = window * (1 + preds)
    dates = df.index[-n_test:]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true_p, label='Actual', linewidth=2)
    plt.plot(dates, y_pred_p, label='Predicted', linewidth=2, alpha=0.8)
    plt.title(f"{sym} – Actual vs Predicted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{sym}_returns_pred.png")
    plt.close()
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true_p, y_pred_p))
    mae = mean_absolute_error(y_true_p, y_pred_p)
    
    # Calculate MAPE safely
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs(np.where(y_true_p != 0, (y_true_p - y_pred_p) / y_true_p, 0))) * 100
    
    print(f"\n{sym} Metrics:")
    print(f" • Directional Accuracy: {dir_acc:.2f}%")
    print(f" • RMSE: {rmse:.2f}")
    print(f" • MAE: {mae:.2f}")
    print(f" • MAPE: {mape:.2f}%")
    
    # Store metrics for ranking
    metrics_data.append({
        'symbol': sym,
        'dir_acc': dir_acc,
        'rmse': rmse,
        'mape': mape
    })

# Convert to DataFrame for easier manipulation
metrics_df = pd.DataFrame(metrics_data)

if not metrics_df.empty:
    # Normalize metrics (convert to z-scores)
    for col in ['dir_acc', 'rmse', 'mape']:
        metrics_df[f'{col}_norm'] = (metrics_df[col] - metrics_df[col].mean()) / metrics_df[col].std()

    # Compute composite score (higher is better)
    # Score = -MAPE% - RMSE_norm + DirAcc%
    metrics_df['score'] = (-metrics_df['mape_norm'] - metrics_df['rmse_norm'] + metrics_df['dir_acc_norm'])

    # Sort by composite score
    metrics_df = metrics_df.sort_values('score', ascending=False)

    print("\n=== Asset Performance Ranking ===")
    print("\nRanked by composite score (higher is better):")
    for idx, row in metrics_df.iterrows():
        print(f"\n{row['symbol']}:")
        print(f" • Composite Score: {row['score']:.2f}")
        print(f" • Directional Accuracy: {row['dir_acc']:.2f}%")
        print(f" • RMSE: {row['rmse']:.2f}")
        print(f" • MAPE: {row['mape']:.2f}%")

print("\nAll done!")
