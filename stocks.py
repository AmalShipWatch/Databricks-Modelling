import yfinance as yf
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm


def fetch_data(ticker, period, interval):
    """
    Fetch historical stock data using yfinance.
    
    Parameters:
    -----------
    ticker : str
        The stock symbol (e.g., 'AAPL').
    period : str
        The time period to fetch (e.g., '60d' for 60 days).
    interval : str
        The data frequency (e.g., '15m' for 15 minutes).
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with the 'Close' price column.
    
    Raises:
    -------
    ValueError
        If the ticker is not found or no data is returned.
    """
    try:
        print(f"Fetching data for {ticker} (period: {period}, interval: {interval})...")
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
        
        # Ensure we have the 'Close' column
        if 'Close' not in data.columns:
            raise ValueError(f"'Close' price column not found in data for {ticker}")
        
        print(f"Successfully fetched {len(data)} data points.")
        return data
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        raise


def preprocess_data(data, time_steps):
    """
    Scale the data and create sequences for the LSTM model.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame from the fetch_data function.
    time_steps : int
        Number of past data points to use for predicting the next one.
    
    Returns:
    --------
    tuple
        (X_train, y_train, scaler) where:
        - X_train: numpy array of shape (n_samples, time_steps, 1)
        - y_train: numpy array of target values
        - scaler: MinMaxScaler object for inverse transformation
    """
    try:
        print(f"Preprocessing data with time_steps={time_steps}...")
        
        # Extract the 'Close' prices
        prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        X_train = []
        y_train = []
        
        for i in range(len(scaled_prices) - time_steps):
            X_train.append(scaled_prices[i:i + time_steps, 0])
            y_train.append(scaled_prices[i + time_steps, 0])
        
        X_train = np.array(X_train).reshape(-1, time_steps, 1)
        y_train = np.array(y_train)
        
        print(f"Preprocessed data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
        return X_train, y_train, scaler
    
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        raise


class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for time series prediction.
    """
    def __init__(self, time_steps, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take the last output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EarlyStopping:
    """
    Simple early stopping for training based on loss improvement.
    """
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def build_or_load_model(time_steps):
    """
    Define and compile the LSTM model or load a pre-existing one.
    
    Parameters:
    -----------
    time_steps : int
        The number of input time steps for the model's input shape.
    
    Returns:
    --------
    nn.Module
        The PyTorch LSTM model.
    """
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'lstm_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}...")
            model = LSTMModel(time_steps)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print("Model loaded successfully.")
        else:
            print("Building new LSTM model...")
            model = LSTMModel(time_steps)
            model.to(device)
            print("Model built successfully.")
        
        return model, device
    
    except Exception as e:
        print(f"Error building or loading model: {str(e)}")
        raise


def run_dynamic_training_loop():
    """
    Run the entire dynamic training process in a continuous loop.
    
    The loop continuously:
    1. Fetches new stock data
    2. Preprocesses the data
    3. Retrains the model with the new data
    4. Makes a prediction for the next price
    5. Waits before the next cycle
    """
    # Define Constants
    TICKER = "AAPL"
    PERIOD = "15d"
    INTERVAL = "15m"
    TIME_STEPS = 180
    TRAINING_INTERVAL_SECONDS = 30  # 1 minute between training cycles
    
    # Initial Setup
    model, device = build_or_load_model(TIME_STEPS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Fetch full historical data once at the start
    print(f"Fetching initial full dataset for {TICKER}...")
    accumulated_data = fetch_data(TICKER, PERIOD, INTERVAL)
    print(f"Initial dataset size: {len(accumulated_data)} rows\n")

    # Infinite Loop
    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"Starting Training Cycle {cycle} at {pd.Timestamp.now()}")
        print(f"{'='*60}\n")
        
        try:
            if cycle > 1:
                # Fetch fresh data and append only new rows to accumulated dataset
                print(f"Fetching fresh data for incremental update...")
                fresh_data = fetch_data(TICKER, PERIOD, INTERVAL)
                try:
                    last_idx = accumulated_data.index[-1]
                    new_rows = fresh_data[fresh_data.index > last_idx]
                except Exception:
                    new_rows = fresh_data[~fresh_data.index.isin(accumulated_data.index)]

                if not new_rows.empty:
                    accumulated_data = pd.concat([accumulated_data, new_rows])
                    print(f"Appended {len(new_rows)} new row(s). Total dataset size: {len(accumulated_data)} rows.")
                else:
                    print(f"No new rows in fresh data. Using accumulated dataset ({len(accumulated_data)} rows).")
            else:
                print(f"Cycle 1: Using initial accumulated dataset with {len(accumulated_data)} rows.")

            # Preprocess the accumulated data
            X_train, y_train, scaler = preprocess_data(accumulated_data, TIME_STEPS)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
            
            # Train the model with incremental learning using tqdm and early stopping
            print(f"Training model on {len(X_train)} samples...")
            model.train()

            max_epochs = 100
            batch_size = 32
            early_stopper = EarlyStopping(patience=10, min_delta=1e-4)
            pbar = tqdm(range(1, max_epochs + 1), desc="Epochs")
            for epoch in pbar:
                epoch_loss = 0.0
                num_batches = 0

                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]

                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches if num_batches else epoch_loss
                pbar.set_postfix({'loss': f"{avg_loss:.6f}"})

                # Early stopping check
                early_stopper.step(avg_loss)
                if early_stopper.should_stop:
                    pbar.set_description(f"Epochs (early stopped at {epoch})")
                    break

            # Save the updated model into the models/ directory
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'lstm_model_{cycle}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # Make a prediction
            print("\nMaking prediction for the next price...")
            model.eval()
            with torch.no_grad():
                last_sequence = accumulated_data['Close'].values[-TIME_STEPS:].reshape(-1, 1)
                last_sequence_scaled = scaler.transform(last_sequence)
                last_sequence_scaled = torch.FloatTensor(last_sequence_scaled).reshape(1, TIME_STEPS, 1).to(device)
                
                # Predict the next scaled price
                next_price_scaled = model(last_sequence_scaled).cpu().numpy()
                
                # Inverse transform to get the actual price
                next_price = scaler.inverse_transform(next_price_scaled)
            
            # Ensure we have plain Python floats for printing/percentage calculations
            def _safe_last_value(close_like):
                # handle pandas Series, numpy arrays (1D or 2D), and lists
                try:
                    # pandas Series or similar
                    return float(close_like.iloc[-1])
                except (AttributeError, TypeError):
                    arr = np.asarray(close_like)
                    if arr.ndim == 0:
                        return float(arr.item())
                    elif arr.ndim == 1:
                        return float(arr[-1])
                    elif arr.ndim == 2 and arr.shape[1] == 1:
                        return float(arr[-1, 0])
                    else:
                        # flatten fallback
                        return float(arr.reshape(-1)[-1])

            current_price = _safe_last_value(accumulated_data['Close'])
            predicted_price = float(next_price[0, 0])
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            print(f"\nCurrent Price: ${current_price:.2f}")
            print(f"Predicted Next Price: ${predicted_price:.2f}")
            print(f"Expected Change: ${price_change:.2f} ({price_change_pct:.2f}%)")
            
            print(f"\n{'='*60}")
            print(f"Training cycle {cycle} completed successfully.")
            print(f"Next training cycle in {TRAINING_INTERVAL_SECONDS} seconds...")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\nError during training cycle {cycle}: {str(e)}")
            print(f"Retrying in {TRAINING_INTERVAL_SECONDS} seconds...\n")
        
        # Wait before the next cycle
        time.sleep(TRAINING_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run_dynamic_training_loop()
    except KeyboardInterrupt:
        print("\n\nTraining loop interrupted by user.")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
