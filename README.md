# RNN Price Predictor

A graphical desktop application that utilizes a Recurrent Neural Network (RNN) to predict financial asset prices based on historical data.

## Description

This tool fetches real-time historical data from Yahoo Finance, processes it with technical analysis indicators (RSI, MACD, Bollinger Bands, etc.), and trains a TensorFlow RNN model locally to predict the End-Of-Day (EOD) close price.

## Features

- **Dark Mode GUI**: Clean, dark-themed interface built with Tkinter.
- **Symbol Selector**: Pre-configured list of popular Crypto, Stocks, and Forex symbols.
- **Real-time Training**: Trains a fresh model on the latest data every time you run a prediction.
- **Technical Analysis**: Automates feature engineering using `pandas_ta`.

## Requirements

Ensure you have Python 3.7+ installed.

### Dependencies

Install the required Python packages using pip. Open your terminal or command prompt and run:

```bash
pip install pandas pandas_ta scikit-learn numpy yfinance tensorflow
```

*Note: `tkinter` is required for the GUI. It is included with standard Python installations on Windows and macOS. On Linux, you might need to install `python3-tk` explicitly via your package manager.*

## How to Run

1. Open your terminal or command prompt.
2. Navigate to the folder containing the script.
3. Run the following command:

   ```bash
   python predict_rnn.py
   ```

## How to Build Executable (.exe)

To create a standalone `.exe` file that can run on any Windows machine without Python installed:

1. Install the requirements (including pyinstaller):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the build script:
   ```bash
   python build_exe.py
   ```
3. Once finished, find your application in the **`dist`** folder.

## How to Use

1. The application window will open with a dark theme.
2. **Select Symbol**: Use the dropdown menu at the top to pick an asset (e.g., `BTC-USD`, `AAPL`, `EURUSD=X`).
3. **Predict**: Click the blue **Predict** button.
4. **Wait**: The model will download data and begin training. You will see the logs, training epochs, and model summary in the main text window.
5. **Result**: Once finished, the specific prediction for today's close price will be displayed at the bottom of the log.

## Disclaimer

**For Educational Purposes Only.**
This application is a demonstration of applying deep learning to time-series data. It is not financial advice. Trading financial markets involves significant risk.
