import plotly.graph_objects as go

# Create a simple line plot
fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17]))

# Update layout
fig.update_layout(title="Simple Plot", xaxis_title="X Axis", yaxis_title="Y Axis")

# Show the figure
fig.show()