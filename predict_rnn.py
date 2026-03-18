import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import sys

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# A curated list of popular symbols. Fetching all symbols from yfinance is not practical.
SYMBOLS = [
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD",
    # Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    # Forex
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDCAD=X", "USDCAD=X",
    # Indices
    "^GSPC", "^IXIC", "^DJI",
    # Commodities
    "GC=F" # Gold
]


# Class to redirect stdout to the GUI
class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)

    def flush(self):
        pass

def run_prediction(symbol, text_widget):
    # Redirect stdout/stderr to the text widget
    sys.stdout = TextRedirector(text_widget)
    
    print(f"Fetching data for {symbol}...")
    try:
        # Fetch 1H data for features and daily data for targets
        df = yf.Ticker(symbol).history(period="60d", interval="1h")
        df_daily = yf.Ticker(symbol).history(period="60d", interval="1d")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty:
        print(f"\nError: No data found for symbol '{symbol}'.")
        print("Please check if the symbol is correct. For Forex, try appending '=X' (e.g., AUDCAD=X).")
        return

    if df_daily.empty:
        print(f"\nError: No daily data found for symbol '{symbol}' to create targets.")
        return


    # --- 1. Data Processing ---
    # Resample 1-hour data to 4-hour candles (H4)
    df = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    df.columns = df.columns.str.lower() # Convert columns to lowercase (open, high, low...)
    df = df.dropna()

    print("--- Initial Data ---")
    print(df.tail())

    # --- 2. Feature Engineering ---
    # Create a rich set of features that the model can learn from.
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.ema(length=50, append=True, col_names='EMA_50')

    # Create custom lagged features
    for i in range(1, 6):
        df[f'price_change_lag_{i}'] = df['close'].diff(i)

    df.dropna(inplace=True)

    # --- 3. Target Definition & Data Splitting ---
    prediction_day_date = df.index.max().normalize()
    historical_df = df[df.index < prediction_day_date].copy()
    prediction_df = df[df.index.normalize() == prediction_day_date].copy()

    if prediction_df.empty:
        print("\nError: No data available for the current day to make a prediction.")
        print("This can happen on market holidays or if the script is run before trading begins.")
        return

    # Create maps from the true daily data for accurate EOD targets
    # Use date objects as keys to ensure robust matching between intraday and daily data
    daily_close_map = df_daily['Close'].groupby(df_daily.index.date).last()
    daily_high_map = df_daily['High'].groupby(df_daily.index.date).max()
    daily_low_map = df_daily['Low'].groupby(df_daily.index.date).min()

    historical_df['target_eod_close'] = historical_df.index.map(lambda x: daily_close_map.get(x.date()))
    historical_df['target_eod_high'] = historical_df.index.map(lambda x: daily_high_map.get(x.date()))
    historical_df['target_eod_low'] = historical_df.index.map(lambda x: daily_low_map.get(x.date()))
    historical_df.dropna(inplace=True)

    # --- 4. Data Preparation for RNN Model ---
    features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 
                           'target_eod_close', 'target_eod_high', 'target_eod_low']
    features = [col for col in historical_df.columns if col not in features_to_exclude]
    X = historical_df[features]
    y = historical_df[['target_eod_close', 'target_eod_high', 'target_eod_low']] # Keep as DataFrame for scaler

    # Split data into training and testing sets before scaling
    train_size = int(len(X) * 0.8)
    X_train_df, X_test_df = X[:train_size], X[train_size:]
    y_train_df, y_test_df = y[:train_size], y[train_size:]

    # Scale features and target
    # Fit scalers ONLY on the training data to prevent data leakage
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_df)
    X_test_scaled = x_scaler.transform(X_test_df)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_df)
    y_test_scaled = y_scaler.transform(y_test_df)

    # Create sequences for the RNN
    sequence_length = 10 # Use the last 10 time steps (4-hour candles) to predict the next

    def create_sequences(X_data, y_data, seq_length):
        """
        Transforms 2D time series data into 3D sequences for RNNs.
        For a sequence from t to t+seq_length-1, the target is y at t+seq_length.
        """
        X_seq, y_seq = [], []
        for i in range(len(X_data) - seq_length):
            X_seq.append(X_data[i:(i + seq_length)])
            y_seq.append(y_data[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

    print(f"\n--- Data Shapes for RNN ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --- 5. Model Training ---
    print("\n--- Training RNN Model ---")
    model = Sequential([
        SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        SimpleRNN(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(3) # Output layer for Close, High, Low
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    # Redirect summary print
    model.summary(print_fn=lambda x: print(x))

    # Train the model
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # --- 6. Evaluation & Prediction ---
    # Evaluate the model on the unseen historical test set
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_inv = y_scaler.inverse_transform(y_test)

    # Calculate metrics for each target
    mae_close = mean_absolute_error(y_test_inv[:, 0], y_pred[:, 0])
    r2_close = r2_score(y_test_inv[:, 0], y_pred[:, 0])
    mae_high = mean_absolute_error(y_test_inv[:, 1], y_pred[:, 1])
    r2_high = r2_score(y_test_inv[:, 1], y_pred[:, 1])
    mae_low = mean_absolute_error(y_test_inv[:, 2], y_pred[:, 2])
    r2_low = r2_score(y_test_inv[:, 2], y_pred[:, 2])

    print("\n--- Model Evaluation on Historical Test Set ---")
    print(f"Close - MAE: {mae_close:.4f}, R²: {r2_close:.4f}")
    print(f"High  - MAE: {mae_high:.4f}, R²: {r2_high:.4f}")
    print(f"Low   - MAE: {mae_low:.4f}, R²: {r2_low:.4f}")
    print(f"(MAE means the EOD predictions were off by an average of these values on historical data)")

    # Predict today's close using the last available sequence of data.
    # We need the last 'sequence_length' data points from the full dataframe.
    full_features_df = df.drop(columns=features_to_exclude, errors='ignore')

    # Get the last `sequence_length` rows
    latest_sequence_df = full_features_df.iloc[-sequence_length:]

    # Scale these features using the already-fitted x_scaler
    latest_sequence_scaled = x_scaler.transform(latest_sequence_df)

    # Reshape for RNN input: (1, sequence_length, n_features)
    input_for_prediction = np.expand_dims(latest_sequence_scaled, axis=0)

    # Make the prediction
    todays_eod_prediction_scaled = model.predict(input_for_prediction)

    # Inverse transform the prediction to get the actual price
    todays_eod_prediction = y_scaler.inverse_transform(todays_eod_prediction_scaled)
    
    pred_close, pred_high, pred_low = todays_eod_prediction[0]

    print("\n--- Prediction for Today's End-of-Day ---")
    last_known_price = prediction_df['close'].iloc[-1]
    print(f"Last known price for today: {last_known_price:.4f}")

    # --- Post-processing and Sanity Checks ---
    # The predicted EOD low cannot be higher than a price that has already occurred today.
    final_low = min(pred_low, last_known_price)
    # The predicted EOD close must be at least the predicted EOD low.
    final_close = max(final_low, pred_close)
    # The predicted EOD high must be at least the predicted EOD close.
    final_high = max(final_close, pred_high)

    print(f"Predicted EOD Close: {final_close:.4f}")
    print(f"Predicted EOD High:  {final_high:.4f}")
    print(f"Predicted EOD Low:   {final_low:.4f}")

    if final_low != pred_low or final_close != pred_close or final_high != pred_high:
        print("(Values adjusted for logical consistency: L <= C <= H and Low <= Last Known Price)")
    
    # Restore stdout
    sys.stdout = sys.__stdout__

def start_gui():
    root = tk.Tk()
    root.title("RNN Price Predictor")
    root.geometry("800x600")
    root.configure(bg="#1e1e1e") # Dark background for the main window

    # --- Top Frame for Controls ---
    frame = tk.Frame(root, bg="#1e1e1e")
    frame.pack(pady=10)
    
    tk.Label(frame, text="Select Symbol:", bg="#1e1e1e", fg="white").pack(side=tk.LEFT, padx=(0, 5))

    # Use a Combobox for symbol selection
    style = ttk.Style()
    style.theme_use('clam') # Use a theme that is easier to configure
    style.configure("TCombobox", fieldbackground="#3c3c3c", background="#3c3c3c", foreground="white", arrowcolor="white")
    style.map('TCombobox', fieldbackground=[('readonly', '#3c3c3c')])

    symbol_var = tk.StringVar()
    symbol_combobox = ttk.Combobox(frame, textvariable=symbol_var, values=SYMBOLS, state='readonly')
    symbol_combobox.set("BTC-USD") # Set default value
    symbol_combobox.pack(side=tk.LEFT, padx=10)

    # --- Output Text Area ---
    text_area = scrolledtext.ScrolledText(
        root, width=90, height=30,
        bg="#121212", fg="lightgray",
        insertbackground="white", # Cursor color
        selectbackground="#0078d7" # Selection color
    )
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    def on_predict():
        symbol = symbol_var.get()
        if not symbol: return
        text_area.delete(1.0, tk.END)
        # Run in a separate thread to keep GUI responsive
        t = threading.Thread(target=run_prediction, args=(symbol, text_area))
        t.daemon = True
        t.start()

    # --- Predict Button ---
    tk.Button(frame, text="Predict", command=on_predict, bg="#0078d7", fg="white", activebackground="#005a9e", activeforeground="white", relief=tk.FLAT, borderwidth=0, padx=10, pady=2).pack(side=tk.LEFT)
    
    root.mainloop()

if __name__ == "__main__":
    start_gui()
