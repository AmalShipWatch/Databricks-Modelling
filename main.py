import torch
import torch.nn as nn
import sqlite3
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(1, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

def fetch_data(imo_value):
    db_path = "/Volumes/workspace/digital_twin/digital-twin/digital_twin.db"
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM preprocessed_noon_data WHERE imo_value = {imo_value}"
    result = conn.execute(query).fetchall()
    columns = [desc[0] for desc in conn.execute(query).description]  # <-- Add this line
    from pyspark.sql.types import StructType, StructField, StringType

    if not result:
        schema = StructType([StructField(col, StringType(), True) for col in columns])
        tables_df = spark.createDataFrame([], schema=schema)
    else:
        tables_df = spark.createDataFrame(result, schema=columns)

    return tables_df


def preprocess_and_split(df, feature_col, target_col, test_ratio=0.2):
    """
    Preprocesses the DataFrame, applies Standard scaling, and splits it into training and testing sets.
    Args:
        df: Spark DataFrame
        feature_col: Name of the feature column
        target_col: Name of the target column
        test_ratio: Fraction of data to use for testing
    Returns:
        X_train, y_train, X_test, y_test (all torch tensors, scaled)
        feature_scaler, target_scaler (fitted scalers)
    """
    pandas_df = df.select(feature_col, target_col).toPandas()
    X = pandas_df[feature_col].values.reshape(-1, 1)
    y = pandas_df[target_col].values.reshape(-1, 1)

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

    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler


def train_model(model, X_train, y_train, epochs=10000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return model, loss_history


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
    Args:
        X_test: Test features (torch tensor, scaled)
        y_test_inv: True targets (inverse transformed)
        predictions_inv: Model predictions (inverse transformed)
        feature_scaler: Fitted StandardScaler for feature
    """
    import plotly.graph_objs as go
    import plotly.io as pio
    X_test_np = X_test.cpu().numpy().flatten().reshape(-1, 1)
    X_test_inv = feature_scaler.inverse_transform(X_test_np).flatten()
    y_test_inv = y_test_inv.flatten()
    pred_inv = predictions_inv.flatten()

    # Sort for line plot
    sort_idx = X_test_inv.argsort()
    X_sorted = X_test_inv[sort_idx]
    pred_sorted = pred_inv[sort_idx]

    trace_true = go.Scatter(x=X_test_inv, y=y_test_inv, mode='markers', name='True Values')
    trace_pred = go.Scatter(x=X_sorted, y=pred_sorted, mode='lines', name='Predicted Values')
    layout = go.Layout(title='Model Fitting: True vs Predicted',
                      xaxis=dict(title='Feature (original scale)'),
                      yaxis=dict(title='Target (original scale)'))
    fig = go.Figure(data=[trace_true, trace_pred], layout=layout)
    pio.show(fig)


# Main workflow example
if __name__ == "__main__":
    # Example values (replace with actual column names and IMO value)
    imo_value = 9935301
    feature_col = "avg_speed_unscaled"
    target_col = "consumption_unscaled"

    # Fetch data
    df = fetch_data(imo_value)

    # Preprocess and split
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = preprocess_and_split(df, feature_col, target_col)

    # Initialize and train model
    model = SimpleNN()
    model, loss_history = train_model(model, X_train, y_train, epochs=10000, lr=0.001)

    # Test and evaluate
    predictions_inv, mse, mape, r2, y_test_inv = test_and_evaluate(model, X_test, y_test, target_scaler)
    print(f"Test MSE (original scale): {mse}")
    print(f"Test MAPE (original scale): {mape:.2f}%")
    print(f"Test R² score (original scale): {r2:.4f}")

    # Plot model fitting
    plot_model_fitting(X_test, y_test_inv, predictions_inv, feature_scaler)

