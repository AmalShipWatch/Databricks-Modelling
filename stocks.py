import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
from tqdm import tqdm

# --- CONFIGURATION ---
TICKER = "BTC-USD"
INTERVAL = "1m"          # <--- CHANGED: Fastest possible setting
TIME_STEPS = 15          # Look back 15 minutes to predict the next minute
TRAINING_INTERVAL = 60   # <--- CHANGED: Wait 60s so a new candle is guaranteed to form
EPOCHS_PER_CYCLE = 5     # Keep training fast so we don't miss the next minute

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# --- 1. DATA FUNCTIONS ---

def fetch_data(period):
    """Fetches data from Yahoo Finance."""
    try:
        df = yf.download(TICKER, period=period, interval=INTERVAL, progress=False)
        if df.empty:
            return None
        # Handle MultiIndex columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Close']] # We only need the Close price
    except Exception as e:
        print(f"API Error: {e}")
        return None

def prepare_tensors(data, time_steps):
    """Scales data and creates (X, y) sequences for LSTM."""
    # Scale data (0 to 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps])
    
    # Convert to PyTorch Tensors
    X = torch.FloatTensor(np.array(X)).to(device)
    y = torch.FloatTensor(np.array(y)).to(device)
    
    return X, y, scaler, scaled_data

# --- 2. MODEL DEFINITION ---

class BitcoinLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(BitcoinLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :]) 
        return out

# --- 3. MAIN EXECUTION LOOP ---

def main():
    # --- INITIALIZATION PHASE ---
    print(f"--- STARTING SYSTEM FOR {TICKER} ---")
    
    # 1. Initialize Model
    model = BitcoinLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 2. Cold Start: Fetch max history (60 days)
    print(f"Fetching initial {INTERVAL} history...")
    master_df = fetch_data(period="60d")
    
    if master_df is None or len(master_df) < TIME_STEPS:
        print("Not enough initial data. Exiting.")
        return

    print(f"Initial data loaded. Total rows: {len(master_df)}")

    # 3. Initial Training
    print("Performing initial training...")
    X_train, y_train, scaler, _ = prepare_tensors(master_df, TIME_STEPS)
    
    model.train()
    pbar = tqdm(range(20), desc="Initial Training")
    total_loss = 0
    for epoch in pbar:
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (epoch + 1)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})
    print(f"Initial Training Complete. Loss: {loss.item():.6f}")

    # --- DYNAMIC LOOP PHASE ---
    print("\n--- ENTERING LIVE LOOP (Updates every 30s) ---")
    
    try:
        while True:
            time.sleep(TRAINING_INTERVAL)
            
            # A. Fetch small batch (Last 1 day)
            new_df = fetch_data(period=INTERVAL)
            
            if new_df is not None:
                # B. Logic: Find rows in new_df that are NEWER than master_df
                last_saved_time = master_df.index[-1]
                new_rows = new_df[new_df.index > last_saved_time]

                if not new_rows.empty:
                    # C. Update Master Data
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] New data detected: {len(new_rows)} rows.")
                    master_df = pd.concat([master_df, new_rows])
                    
                    # D. Re-Process Data (Re-scale with new range)
                    X_new, y_new, scaler, scaled_full = prepare_tensors(master_df, TIME_STEPS)
                    
                    # E. Incremental Training (Fine-tuning)
                    model.train()
                    for _ in range(EPOCHS_PER_CYCLE):
                        optimizer.zero_grad()
                        outputs = model(X_new)
                        loss = criterion(outputs, y_new)
                        loss.backward()
                        optimizer.step()
                    
                    # F. Prediction
                    model.eval()
                    with torch.no_grad():
                        # Get last sequence from the updated data
                        last_seq = torch.FloatTensor(scaled_full[-TIME_STEPS:]).unsqueeze(0).to(device)
                        pred_scaled = model(last_seq).cpu().numpy()
                        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                        
                        current_price = master_df['Close'].iloc[-1]
                        
                        print(f" -> Current BTC: ${current_price:.2f}")
                        print(f" -> Predicted Next Close: ${pred_price:.2f}")
                        print(f" -> Loss: {loss.item():.6f}")
                
                else:
                    # No new candle yet (still within the 15m block)
                    # Optional: You can print a 'heartbeat' or just pass
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No new candle yet. Waiting...")

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()