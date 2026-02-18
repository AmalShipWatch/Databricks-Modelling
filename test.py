import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Generate dummy stock market data
np.random.seed(42)
days = 365
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(days)]

# Create realistic stock price data with trend and noise
base_price = 100
trend = np.linspace(0, 30, days)
seasonality = 10 * np.sin(np.linspace(0, 4 * np.pi, days))
noise = np.random.normal(0, 5, days)
prices = base_price + trend + seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Price': prices
})

# Prepare data for forecasting
X = np.arange(len(df)).reshape(-1, 1)
y = df['Price'].values

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Generate forecast for next 60 days
forecast_days = 60
future_X = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
forecast_prices = model.predict(future_X)

# Add some noise to make forecast more realistic
forecast_noise = np.random.normal(0, 3, forecast_days)
forecast_prices_noisy = forecast_prices + forecast_noise

# Calculate confidence intervals (simple approach)
std_dev = np.std(y - model.predict(X))
confidence_interval = 1.96 * std_dev  # 95% confidence

upper_bound = forecast_prices + confidence_interval
lower_bound = forecast_prices - confidence_interval

# Create forecast dates
last_date = pd.Timestamp(dates[-1]).to_pydatetime()
forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

# Create interactive Plotly figure
fig = go.Figure()

# Add historical prices
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Price'],
    mode='lines',
    name='Historical Prices',
    line=dict(color='blue', width=2),
    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
))

# Add forecasted prices
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=forecast_prices_noisy,
    mode='lines',
    name='Forecasted Prices',
    line=dict(color='red', width=2, dash='dash'),
    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: $%{y:.2f}<extra></extra>'
))

# Add upper confidence bound
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=upper_bound,
    mode='lines',
    name='Upper Bound (95% CI)',
    line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
    showlegend=True,
    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Upper</b>: $%{y:.2f}<extra></extra>'
))

# Add lower confidence bound
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=lower_bound,
    mode='lines',
    name='Lower Bound (95% CI)',
    line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
    fill='tonexty',
    fillcolor='rgba(255, 0, 0, 0.1)',
    showlegend=True,
    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Lower</b>: $%{y:.2f}<extra></extra>'
))

# Add vertical line to mark forecast start
fig.add_vline(
    x=dates[-1],
    line_dash="dot",
    line_color="green",
    annotation_text="Forecast Start",
    annotation_position="top"
)

# Update layout for better interactivity
fig.update_layout(
    title={
        'text': 'Stock Market Price Forecasting - Interactive Dashboard',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial, sans-serif'}
    },
    xaxis_title='Date',
    yaxis_title='Stock Price ($)',
    hovermode='x unified',
    template='plotly_white',
    width=1200,
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        rangeslider=dict(visible=True),
        type='date'
    )
)

# Add range selector buttons
fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all", label="All")
        ])
    )
)

# Display statistics
print("=" * 60)
print("STOCK MARKET FORECASTING ANALYSIS")
print("=" * 60)
print(f"\nHistorical Data Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"Forecast Period: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
print(f"\nCurrent Price: ${prices[-1]:.2f}")
print(f"Forecasted Price (60 days): ${forecast_prices_noisy[-1]:.2f}")
print(f"Expected Change: ${forecast_prices_noisy[-1] - prices[-1]:.2f} ({((forecast_prices_noisy[-1] - prices[-1]) / prices[-1] * 100):.2f}%)")
print(f"\nModel R² Score: {model.score(X, y):.4f}")
print(f"95% Confidence Interval: ±${confidence_interval:.2f}")
print("=" * 60)

# Show the figure
fig.show()