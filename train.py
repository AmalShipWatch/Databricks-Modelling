import torch
import torch.nn as nn
import sqlite3
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score
from pyspark.sql import SparkSession

# Initialize Spark Session
# spark = SparkSession.builder.appName("TrainModel").getOrCreate()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleNN(nn.Module):
    def __init__(self, input_size=2):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

def fetch_data(imo_value):
    db_path = "database/digital_twin.db"
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM preprocessed_sensor_data WHERE imo_value = {imo_value}"
    
    tables_df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nFetched {len(tables_df)} rows for IMO value {imo_value}")
    return tables_df

def preprocess_and_split(df, feature_cols, target_col, test_ratio=0.2):
    """
    Preprocesses the DataFrame, applies Standard scaling, and splits it into training and testing sets.
    Args:
        df: Spark DataFrame
        feature_cols: List of feature column names or single string
        target_col: Name of the target column
        test_ratio: Fraction of data to use for testing
    Returns:
        X_train, y_train, X_test, y_test (all torch tensors, scaled)
        X_train_raw, X_test_raw, y_train_raw, y_test_raw (unscaled numpy arrays)
        feature_scaler, target_scaler (fitted scalers)
    
    Raises:
        ValueError: If columns don't exist or data is invalid
    """
    # Handle single string or list of columns
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    
    # Validate that all required columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Validate that we have features
    if len(feature_cols) == 0:
        raise ValueError("feature_cols cannot be empty")
    
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # Validate data
    if X.shape[0] == 0:
        raise ValueError("DataFrame is empty")
    
    if np.isnan(X).any() or np.isnan(y).any():
        print("⚠️  Warning: NaN values found in data. Removing NaN rows...")
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X = X[mask]
        y = y[mask]
        if len(X) == 0:
            raise ValueError("All data was NaN - cannot proceed")
    
    print(f"✓ Data shape: {X.shape} ({len(feature_cols)} features, {X.shape[0]} samples)")

    # Shuffle and split
    num_samples = X.shape[0]
    indices = torch.randperm(num_samples)
    test_size = int(num_samples * test_ratio)
    test_indices = indices[:test_size].numpy()
    train_indices = indices[test_size:].numpy()
    X_train_raw, y_train_raw = X[train_indices], y[train_indices]
    X_test_raw, y_test_raw = X[test_indices], y[test_indices]

    # Fit scalers on training data
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train_raw)
    y_train = target_scaler.fit_transform(y_train_raw)
    X_test = feature_scaler.transform(X_test_raw)
    y_test = target_scaler.transform(y_test_raw)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    print(f"✓ Train/test split: {len(X_train_raw)}/{len(X_test_raw)} samples")
    print(f"✓ Feature scaler fitted on {feature_scaler.n_features_in_} features")

    return X_train, y_train, X_test, y_test, X_train_raw, X_test_raw, y_train_raw, y_test_raw, feature_scaler, target_scaler


def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=10000, lr=0.001, patience=500):
    """
    Train model with optional early stopping.
    
    Args:
        model: Neural network model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (for early stopping)
        y_val: Validation targets (for early stopping)
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Number of epochs with no improvement before stopping (default 500)
    
    Returns:
        model: Trained model
        loss_history: List of training losses
        best_epoch: Epoch with best validation loss (if early stopping used)
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        # Early stopping: evaluate on validation set
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_loss_history.append(val_loss.item())
            
            # Check if validation loss improved
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                epochs_no_improve = 0
                best_epoch = epoch
            else:
                epochs_no_improve += 1
            
            # Early stopping condition
            if epochs_no_improve >= patience:
                print(f"\n✓ Early stopping at epoch {epoch} (Best epoch: {best_epoch}, Patience: {patience})")
                print(f"  Best validation loss: {best_val_loss:.6f}")
                break
    
    return model, loss_history, best_epoch


def test_and_evaluate(model, X_test, y_test, target_scaler):
    """
    Tests the trained model and evaluates performance using MSE, MAPE, and R² (inverse transformed).
    Args:
        model: Trained SimpleNN instance
        X_test: Test features (torch tensor)
        y_test: Test targets (torch tensor)
        target_scaler: Fitted StandardScaler for target
    Returns:
        predictions: Model predictions (torch tensor, inverse transformed)
        mse: Mean squared error (float, original scale)
        mape: Mean absolute percentage error (float, original scale)
        r2: R² score (float, original scale)
        y_test_inv: True targets (inverse transformed)
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        # Inverse transform
        predictions_inv = target_scaler.inverse_transform(predictions)
        y_test_inv = target_scaler.inverse_transform(y_test_np)
        mse = ((predictions_inv - y_test_inv) ** 2).mean()
        mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
        r2 = r2_score(y_test_inv, predictions_inv)
    return predictions_inv, mse, mape, r2, y_test_inv


def plot_model_fitting(X_test, y_test_inv, predictions_inv, feature_scaler):
    """
    Plots the model fitting using Plotly (original scale).
    Handles both single and multi-feature inputs.
    
    Args:
        X_test: Test features (torch tensor, scaled)
        y_test_inv: True targets (inverse transformed)
        predictions_inv: Model predictions (inverse transformed)
        feature_scaler: Fitted StandardScaler for features
    """
    import plotly.graph_objs as go
    import plotly.io as pio
    
    X_test_np = X_test.cpu().numpy()
    
    # Inverse transform features
    X_test_inv = feature_scaler.inverse_transform(X_test_np)
    
    y_test_inv_flat = y_test_inv.flatten()
    pred_inv_flat = predictions_inv.flatten()
    
    n_features = X_test_inv.shape[1]
    
    if n_features == 1:
        # Single feature: simple scatter + line plot
        X_plot = X_test_inv[:, 0]
        
        # Sort for line plot
        sort_idx = X_plot.argsort()
        X_sorted = X_plot[sort_idx]
        pred_sorted = pred_inv_flat[sort_idx]
        
        trace_true = go.Scatter(x=X_plot, y=y_test_inv_flat, mode='markers', name='True Values')
        trace_pred = go.Scatter(x=X_sorted, y=pred_sorted, mode='lines', name='Predicted Values')
        layout = go.Layout(
            title='Model Fitting: True vs Predicted (Single Feature)',
            xaxis=dict(title='Feature (original scale)'),
            yaxis=dict(title='Target (original scale)')
        )
        fig = go.Figure(data=[trace_true, trace_pred], layout=layout)
    elif n_features == 2:
        # Two features: 3D scatter plot
        X0 = X_test_inv[:, 0]
        X1 = X_test_inv[:, 1]
        
        trace_true = go.Scatter3d(
            x=X0, y=X1, z=y_test_inv_flat,
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='True Values'
        )
        trace_pred = go.Scatter3d(
            x=X0, y=X1, z=pred_inv_flat,
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Predicted Values'
        )
        layout = go.Layout(
            title='Model Fitting: True vs Predicted (Two Features)',
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Target'
            )
        )
        fig = go.Figure(data=[trace_true, trace_pred], layout=layout)
    else:
        # Multiple features: use first two for visualization
        print(f"Warning: Model has {n_features} features. Plotting first 2 features only.")
        X0 = X_test_inv[:, 0]
        X1 = X_test_inv[:, 1]
        
        trace_true = go.Scatter3d(
            x=X0, y=X1, z=y_test_inv_flat,
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='True Values'
        )
        trace_pred = go.Scatter3d(
            x=X0, y=X1, z=pred_inv_flat,
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Predicted Values'
        )
        layout = go.Layout(
            title=f'Model Fitting: True vs Predicted (First 2 of {n_features} Features)',
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Target'
            )
        )
        fig = go.Figure(data=[trace_true, trace_pred], layout=layout)
    
    fig.show(fig)


def test_single_input(model, feature_scaler, target_scaler, X_test_raw=None, manual_input=None):
    """
    Tests a single input value (unscaled) and returns the model's prediction (unscaled).
    
    Args:
        model: Trained SimpleNN instance
        feature_scaler: Fitted StandardScaler for features
        target_scaler: Fitted StandardScaler for target
        X_test_raw: Raw (unscaled) test features as numpy array (optional, for random selection)
        manual_input: Manual input value(s) to test - can be a scalar or list/array (optional, overrides random selection)
    
    Returns:
        input_value: The input value(s) used (unscaled)
        prediction: The model's prediction (unscaled)
    """
    model.eval()
    
    # Get number of features from scaler
    n_features = feature_scaler.n_features_in_
    
    if manual_input is not None:
        # Use manual input
        if isinstance(manual_input, (int, float)):
            manual_input = [manual_input]
        manual_input = np.array(manual_input).flatten()
        
        if len(manual_input) != n_features:
            raise ValueError(
                f"manual_input has {len(manual_input)} features, "
                f"but model expects {n_features} features. "
                f"Expected: {n_features} value(s)."
            )
        input_value = manual_input
    elif X_test_raw is not None:
        # Select random value from test data
        if X_test_raw.ndim != 2 or X_test_raw.shape[1] != n_features:
            raise ValueError(
                f"X_test_raw shape {X_test_raw.shape} is invalid. "
                f"Expected (n_samples, {n_features})"
            )
        random_idx = np.random.randint(0, len(X_test_raw))
        input_value = X_test_raw[random_idx, :]
    else:
        raise ValueError("Either 'X_test_raw' or 'manual_input' must be provided.")
    
    # Scale the input - ensure it's 2D array with correct shape
    input_scaled = feature_scaler.transform(input_value.reshape(1, -1))
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(input_tensor).cpu().numpy()
    
    # Inverse transform the prediction
    prediction = target_scaler.inverse_transform(prediction_scaled)[0, 0]
    
    if n_features == 1:
        print(f"Input value (unscaled): {input_value[0]:.4f}")
    else:
        print(f"Input values (unscaled): {input_value}")
    print(f"Predicted value (unscaled): {prediction:.4f}")
    
    return input_value, prediction


# Main workflow example
if __name__ == "__main__":
    # Example values (replace with actual column names and IMO value)
    imo_value = 9853228
    feature_col1 = "avg_speed_unscaled"
    feature_col2 = "mean_draft_unscaled"
    target_col = "ground_truth"

    # Fetch data
    df = fetch_data(imo_value)

    # Preprocess and split
    X_train, y_train, X_test, y_test, X_train_raw, X_test_raw, y_train_raw, y_test_raw, feature_scaler, target_scaler = preprocess_and_split(df, [feature_col1, feature_col2], target_col)

    # Initialize and train model (input_size=2 for 2 features)
    model = SimpleNN(input_size=len([feature_col1, feature_col2]))
    model, loss_history, best_epoch = train_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.001)

    # Test and evaluate
    predictions_inv, mse, mape, r2, y_test_inv = test_and_evaluate(model, X_test, y_test, target_scaler)
    print(f"\n{'='*50}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Test MSE (original scale): {mse:.6f}")
    print(f"Test MAPE (original scale): {mape:.2f}%")
    print(f"Test R² score (original scale): {r2:.4f}")

    # Plot model fitting (uncomment to visualize)
    plot_model_fitting(X_test, y_test_inv, predictions_inv, feature_scaler)

    # Test single input with random value from test data
    print("\n" + "="*50)
    print("Testing with random data point from test set:")
    print("="*50)
    test_single_input(model, feature_scaler, target_scaler, X_test_raw=X_test_raw)

    # Test single input with manual values (MUST match number of features!)
    print("\n" + "="*50)
    print("Testing with manual input (2 features required):")
    print("="*50)
    # Example: feature1=14.17, feature2=8.5
    test_single_input(model, feature_scaler, target_scaler, manual_input=[14.17, 8.5])

    # Another example
    print("\n" + "="*50)
    print("Testing with another manual input:")
    print("="*50)
    # Example: feature1=10.0, feature2=6.0
    test_single_input(model, feature_scaler, target_scaler, manual_input=[10.0, 6.0])

