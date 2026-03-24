import torch
import torch.nn as nn
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pickle
import os
from tqdm import tqdm
import mlflow
from mlflow.models import infer_signature

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


def save_scaler(target_scaler, save_path="models/scaler.pkl"):
    """
    Saves the target scaler to a pickle file for later use in API serving.
    Args:
        target_scaler: Fitted StandardScaler for target variable
        save_path: Path to save the scaler pickle file
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    
    print(f"✓ Scaler saved to {save_path}")


class ConsumptionPredictor(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper that handles scaling and inverse-scaling.
    Ensures input → scale → model → inverse-scale → output
    """
    def load_context(self, context):
        """
        Load model and scalers from artifacts.
        """
        import torch
        
        # Load model state dict
        self.model = SimpleNN()
        model_state = torch.load(context.artifacts["pytorch_model"])
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        # Load scalers
        self.feature_scaler = pickle.load(open(context.artifacts["feature_scaler"], "rb"))
        self.target_scaler = pickle.load(open(context.artifacts["target_scaler"], "rb"))
    
    def predict(self, context, model_input):
        """
        End-to-end prediction with proper scaling.
        
        Args:
            model_input: numpy array of shape (n_samples, 1) - raw feature values
            
        Returns:
            numpy array of predictions in original consumption units (tonnes)
        """
        import torch
        
        # Step 1: Scale input (features) - same as training
        scaled_input = self.feature_scaler.transform(model_input)
        
        # Step 2: Run model inference
        tensor_input = torch.FloatTensor(scaled_input)
        with torch.no_grad():
            scaled_output = self.model(tensor_input).numpy()
        
        # Step 3: Inverse transform output - convert back to real consumption values
        real_output = self.target_scaler.inverse_transform(scaled_output)
        
        return real_output


def log_model_mlflow(model, X_test_raw, y_test_inv, predictions_inv, target_scaler, feature_scaler,
                     imo_value, feature_col, target_col, epochs=10000):
    """
    Logs model, metrics, and scaler artifacts to MLflow using PyFunc wrapper.
    
    PyFunc wrapper ensures:
    1. Input is scaled (matches training)
    2. Model runs inference
    3. Output is inverse-transformed to real consumption tonnes
    
    Args:
        model: Trained SimpleNN model
        X_test_raw: Raw test features (numpy array)
        y_test_inv: Inverse-transformed actual test targets
        predictions_inv: Inverse-transformed model predictions
        target_scaler: Fitted StandardScaler for target variable
        feature_scaler: Fitted StandardScaler for feature variable
        imo_value: Ship IMO value for tracking
        feature_col: Feature column name
        target_col: Target column name
        epochs: Number of training epochs
    """
    try:
        # Calculate metrics in original scale
        mse = ((predictions_inv - y_test_inv) ** 2).mean()
        mae = mean_absolute_error(y_test_inv, predictions_inv)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_inv, predictions_inv)
        consumption_tonne = mae  # MAE rescaled to original consumption units
        
        # ===== LOG PARAMETERS =====
        mlflow.log_params({
            "epochs": epochs,
            "hidden_layers": 16,
            "output_layer": 1,
            "imo_value": imo_value,
            "feature_col": feature_col,
            "target_col": target_col
        })
        
        # ===== LOG METRICS =====
        mlflow.log_metrics({
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "consumption_tonne": consumption_tonne
        })
        
        # ===== SAVE ARTIFACTS FOR PYFUNC =====
        # Save PyTorch model state dict
        model_path = "pytorch_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save feature scaler
        feature_scaler_path = "feature_scaler.pkl"
        with open(feature_scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)
        
        # Save target scaler
        target_scaler_path = "target_scaler.pkl"
        with open(target_scaler_path, 'wb') as f:
            pickle.dump(target_scaler, f)
        
        # ===== CREATE MODEL SIGNATURE =====
        # Signature uses RAW input (what user provides) and RAW output (what they get back)
        signature = infer_signature(X_test_raw, predictions_inv)
        
        # ===== LOG PYFUNC MODEL WITH WRAPPER =====
        # PyFunc wrapper will handle all scaling automatically
        model_info = mlflow.pyfunc.log_model(
            "consumption_predictor",
            python_model=ConsumptionPredictor(),
            artifacts={
                "pytorch_model": model_path,
                "feature_scaler": feature_scaler_path,
                "target_scaler": target_scaler_path
            },
            signature=signature,
            input_example=X_test_raw[:1],
            registered_model_name="workspace.default.ConsumptionPrediction"
        )
        
        # Cleanup temporary files
        os.remove(model_path)
        os.remove(feature_scaler_path)
        os.remove(target_scaler_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Model logged to MLflow (PyFunc wrapper)")
        print(f"  Input: Raw feature values (e.g., 14.169)")
        print(f"  Processing: Auto-scales → Model → Auto-inverse-scales")
        print(f"  Output: Real consumption tonnes")
        print(f"  Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
        print(f"  R² Score: {r2:.6f}")
        print(f"  Consumption MAE (tonnes): {consumption_tonne:.4f}")
        print(f"✓ Model registered: workspace.default.ConsumptionPrediction")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Warning: MLflow logging failed: {e}")
        import traceback
        traceback.print_exc()
        print("Training completed but metrics were not logged.")


# Main workflow example
if __name__ == "__main__":
    # Example values (replace with actual column names and IMO value)
    imo_value = 9935301
    feature_col = "avg_speed_unscaled"
    target_col = "consumption_unscaled"
    epochs = 10000

    # ===== SET MLFLOW EXPERIMENT =====
    mlflow.set_experiment("/Users/amalmuhammed6677@gmail.com/ship_consumption_model")
    
    # ===== START MLFLOW RUN =====
    with mlflow.start_run():
        # Fetch data
        df = fetch_data(imo_value)

        # Preprocess and split
        X_train, y_train, X_test, y_test, feature_scaler, target_scaler = preprocess_and_split(df, feature_col, target_col)
        
        # Get raw test data before torch conversion (for MLflow signature)
        pandas_df = df.select(feature_col, target_col).toPandas()
        X_raw = pandas_df[feature_col].values.reshape(-1, 1)
        test_size = int(len(X_raw) * 0.2)
        X_test_raw = X_raw[-test_size:].astype(np.float32)  # Convert to float32 for MLflow signature consistency

        # Initialize and train model
        model = SimpleNN()
        model, loss_history = train_model(model, X_train, y_train, epochs=epochs, lr=0.001)

        # Test and evaluate
        predictions_inv, mse, mape, r2, y_test_inv = test_and_evaluate(model, X_test, y_test, target_scaler)
        print(f"\nTest Metrics (original scale):")
        print(f"  MSE: {mse}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R² score: {r2:.4f}")
        
        # Save scaler locally for API serving
        save_scaler(target_scaler, save_path="models/scaler.pkl")
        
        # Log model and metrics to MLflow
        log_model_mlflow(
            model=model,
            X_test_raw=X_test_raw,
            y_test_inv=y_test_inv,
            predictions_inv=predictions_inv,
            target_scaler=target_scaler,
            feature_scaler=feature_scaler,
            imo_value=imo_value,
            feature_col=feature_col,
            target_col=target_col,
            epochs=epochs
        )
