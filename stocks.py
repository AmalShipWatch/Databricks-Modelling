import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
from datetime import datetime
from tqdm import tqdm
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

# --- TUNED CONFIGURATION ---
TICKER = "BTC-USD"
INTERVAL = "1m"           # 1 Minute candles (Fastest available)
TIME_STEPS = 15           # Look back 15 minutes
TRAINING_INTERVAL = 60    # Wait 60s to ensure a new candle forms
EPOCHS_PER_CYCLE = 100      # Quick update

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# --- 1. DATA FUNCTIONS ---

def fetch_data(period):
    """Fetches data from Yahoo Finance."""
    try:
        # For 1m data, max period is '7d'. For others, it can be '60d'
        safe_period = "5d" if INTERVAL == "1m" and period == "60d" else period
        
        df = yf.download(TICKER, period=safe_period, interval=INTERVAL, progress=False)
        
        if df.empty:
            return None
        
        # Clean up MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df[['Close']]
    except Exception as e:
        print(f"API Error: {e}")
        return None

def prepare_tensors(data, time_steps):
    """Scales data and creates (X, y) sequences for LSTM."""
    if len(data) <= time_steps:
        return None, None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps])
    
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
        out = self.fc(out[:, -1, :]) 
        return out

# --- 3. EVALUATION METRICS ---

def evaluate_model(model, X_test, y_test, scaler, scaled_data, time_steps):
    """
    Evaluates model performance using multiple metrics.
    Returns: MSE, MAE, R² score, and predictions
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_np, predictions)
    mae = mean_absolute_error(y_test_np, predictions)
    r2 = r2_score(y_test_np, predictions)
    rmse = np.sqrt(mse)
    # Inverse transform to get actual prices
    pred_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test_np)
    
    # Calculate price-based MAE
    price_mae = mean_absolute_error(actual_prices, pred_prices)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'price_mae': price_mae,
        'rmse': rmse,
        'predictions': predictions,
        'actual': y_test_np
    }
    
    return metrics, y_test_np

def save_model(model, cycle_num, models_dir="models"):
    """
    Saves model to the models directory with cycle number in filename.
    Creates the directory if it doesn't exist.
    """
    # Ensure models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    filename = f"{models_dir}/model_cycle_{cycle_num}.pth"
    
    torch.save(model.state_dict(), filename)
    print(f"Model saved: {filename}")

def log_model_mlflow(model, cycle_num, metrics, X_train=None, pred_scaled=None, best_price_mae=None):
    """
    Logs model and metrics to MLflow.
    Updates 'best' alias only if current model has better (lower) price_mae.
    Returns: updated best_price_mae value
    """
    mlflow.set_experiment('/Users/amalmuhammed6677@gmail.com/bitcoin_forecast')
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "cycle": cycle_num,
            "epochs": EPOCHS_PER_CYCLE,
            "time_steps": TIME_STEPS
        })
        
        # Log metrics
        mlflow.log_metrics({
            "mse": metrics['mse'],
            "mae": metrics['mae'],
            "rmse": metrics['rmse'],
            "r2": metrics['r2'],
            "price_mae": metrics['price_mae']
        })
        
        # Create signature for model input/output
        X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
        pred_scaled_np = pred_scaled.cpu().numpy() if torch.is_tensor(pred_scaled) else pred_scaled
        signature = infer_signature(X_train_np, pred_scaled_np)
        
        # Log PyTorch model
        model_info = mlflow.pytorch.log_model(
            model,
            "bitcoin_model",
            signature=signature,
            input_example=X_train_np[:1],
            registered_model_name="workspace.default.BitcoinForecast"
        )
        
        print(f"Model logged: {model_info.model_uri}")
        print(f"Cycle {cycle_num} - RMSE: {metrics['rmse']:.4f}, Price MAE: ${metrics['price_mae']:.2f}")
        
        # Update best model alias if this is the best so far
        if best_price_mae is None or metrics['price_mae'] < best_price_mae:
            client = MlflowClient()
            try:
                client.set_registered_model_alias("workspace.default.BitcoinForecast", "best", cycle_num)
                print(f"✓ New best model! Price MAE: ${metrics['price_mae']:.2f}")
                return metrics['price_mae']  # Return new best price_mae
            except Exception as e:
                print(f"Warning: Could not set alias: {e}")
                return best_price_mae
        else:
            print(f"✓ Model logged (Not better than best: ${best_price_mae:.2f})")
            return best_price_mae

def deployment_validation():
    """
    Loads and validates the best model from MLflow.
    """
    try:
        model_uri = "models:/workspace.default.BitcoinForecast@best"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        print(f"✓ Best model loaded successfully: {model_uri}")
        return loaded_model
    except Exception as e:
        print(f"Error: Could not load best model: {e}")
        return None



def train_model(model, X_data, y_data, criterion, optimizer, training_type="initial"):
    """
    Unified training function for both initial and incremental training.
    training_type: "initial" or "incremental"
    """
    model.train()
    desc = "Initial Training" if training_type == "initial" else "Incremental Training"
    pbar = tqdm(range(EPOCHS_PER_CYCLE), desc=desc)
    total_loss = 0
    
    for epoch in pbar:
        optimizer.zero_grad()
        outputs = model(X_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (epoch + 1)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})
    
    print(f"{desc} Complete. Loss: {loss.item():.6f}")
    return loss.item()

# --- 4. MAIN EXECUTION LOOP ---

def main():
    print(f"--- STARTING HIGH FREQUENCY LOOP FOR {TICKER} ---")
    
    # 1. Initialize Model
    model = BitcoinLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track best model based on price_mae
    best_price_mae = None
    best_cycle = 0

    # 2. Cold Start
    print("Fetching initial history...")
    master_df = fetch_data(period="60d") # Will auto-adjust to 5d for 1m interval
    
    if master_df is None or len(master_df) < TIME_STEPS:
        print("Not enough initial data. Exiting.")
        return

    print(f"Initial data loaded. Total rows: {len(master_df)}")

    # 3. Initial Training
    print("Performing initial training...")
    X_train, y_train, scaler, _ = prepare_tensors(master_df, TIME_STEPS)
    
    train_model(model, X_train, y_train, criterion, optimizer, training_type="initial")
    
    # Evaluate initial model
    metrics, y_pred = evaluate_model(model, X_train, y_train, scaler, X_train.cpu().numpy(), TIME_STEPS)
    print(f"Initial Model Metrics -> MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.4f}, Price MAE: ${metrics['price_mae']:.2f}")
    
    # Save initial model and track best
    best_price_mae = log_model_mlflow(model, cycle_num=0, metrics=metrics, X_train=X_train, pred_scaled=y_pred, best_price_mae=best_price_mae)
    best_cycle = 0
    save_model(model, cycle_num=0)

    # --- DYNAMIC LOOP PHASE ---
    print(f"\n--- ENTERING LIVE LOOP (Updates every {TRAINING_INTERVAL}s) ---")
    
    cycle_count = 1
    try:
        while cycle_count <= 5:  # Limit to 5 cycles for testing
            # Wait for the next candle
            time.sleep(TRAINING_INTERVAL)
            
            # A. Fetch small batch (Last 1 day is enough to catch new minutes)
            new_df = fetch_data(period="1d")
            
            if new_df is not None:
                # B. Logic: Find rows in new_df that are NEWER than master_df
                last_saved_time = master_df.index[-1]
                new_rows = new_df[new_df.index > last_saved_time]

                if not new_rows.empty:
                    # C. Update Master Data
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] NEW CANDLE! Added {len(new_rows)} row(s).")
                    master_df = pd.concat([master_df, new_rows])
                    print(f"Total rows after Update: {len(master_df)}")
                    
                    # D. Re-Process
                    X_new, y_new, scaler, scaled_full = prepare_tensors(master_df, TIME_STEPS)
                    
                    # E. Incremental Training
                    train_model(model, X_new, y_new, criterion, optimizer, training_type="incremental")
                    
                    # Evaluate model after training
                    metrics, y_pred = evaluate_model(model, X_new, y_new, scaler, X_new.cpu().numpy(), TIME_STEPS)
                    print(f"Cycle {cycle_count} Metrics -> MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.4f}, Price MAE: ${metrics['price_mae']:.2f}")
                    
                    # Save model and update best model tracking
                    best_price_mae = log_model_mlflow(model, cycle_num=cycle_count, metrics=metrics, X_train=X_new, pred_scaled=y_pred, best_price_mae=best_price_mae)
                    if best_price_mae == metrics['price_mae']:
                        best_cycle = cycle_count
                    save_model(model, cycle_num=cycle_count)
                    cycle_count += 1

                    # F. Prediction
                    model.eval()
                    with torch.no_grad():
                        last_seq = torch.FloatTensor(scaled_full[-TIME_STEPS:]).unsqueeze(0).to(device)
                        pred_scaled = model(last_seq).cpu().numpy()
                        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                        current_price = master_df['Close'].iloc[-1]
                        
                        print(f" -> Current: ${current_price:.2f}")
                        print(f" -> Predicted Next 1m: ${pred_price:.2f}")
                
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No new candle yet. (Yahoo updates every 60s)")

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()