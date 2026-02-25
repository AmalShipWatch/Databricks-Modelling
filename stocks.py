import yfinance as yf
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import os


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
    model_path = 'lstm_model.pth'
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
    PERIOD = "60d"
    INTERVAL = "15m"
    TIME_STEPS = 60
    TRAINING_INTERVAL_SECONDS = 3600  # 1 hour between training cycles
    
    # Initial Setup
    model, device = build_or_load_model(TIME_STEPS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Infinite Loop
    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"Starting Training Cycle {cycle} at {pd.Timestamp.now()}")
        print(f"{'='*60}\n")
        
        try:
            # Fetch the latest data
            data = fetch_data(TICKER, PERIOD, INTERVAL)
            
            # Preprocess the data
            X_train, y_train, scaler = preprocess_data(data, TIME_STEPS)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
            
            # Train the model with incremental learning
            print(f"Training model on {len(X_train)} samples...")
            model.train()
            
            for epoch in range(2):
                # Mini-batch training
                batch_size = 32
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
                
                print(f"Epoch {epoch+1}/2 - Loss: {loss.item():.6f}")
            
            # Save the updated model
            model_path = 'lstm_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # Make a prediction
            print("\nMaking prediction for the next price...")
            model.eval()
            with torch.no_grad():
                last_sequence = data['Close'].values[-TIME_STEPS:].reshape(-1, 1)
                last_sequence_scaled = scaler.transform(last_sequence)
                last_sequence_scaled = torch.FloatTensor(last_sequence_scaled).reshape(1, TIME_STEPS, 1).to(device)
                
                # Predict the next scaled price
                next_price_scaled = model(last_sequence_scaled).cpu().numpy()
                
                # Inverse transform to get the actual price
                next_price = scaler.inverse_transform(next_price_scaled)
            
            current_price = data['Close'].iloc[-1]
            predicted_price = next_price[0, 0]
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
