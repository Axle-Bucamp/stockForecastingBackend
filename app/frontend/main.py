import dash
from dash import dcc, html, Input, Output, State
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from io import StringIO

# --- Configuration API ---
host = os.getenv("API_HOST", "localhost")
port = os.getenv("API_PORT", "7060")
API_BASE_URL = f"http://{host}:{port}"

# --- Initialise Dash Application ---
app = dash.Dash(__name__)
server = app.server

# --- Function to Load Tickers ---
def load_tickers():
    try:
        resp = requests.get(f"{API_BASE_URL}/attribute/ticker/list")
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"Erreur de chargement des tickers: {e}")
        return []

# --- Function to Load Stock Data ---
def load_stock_data(ticker):
    try:
        resp = requests.get(f"{API_BASE_URL}/api/json/stock/{ticker}")
        if resp.status_code == 200:
            data = StringIO(resp.json())
            return pd.read_json(data, orient="records")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur de chargement des données: {e}")
        return pd.DataFrame()

# --- Dash Layout with Two Separate Charts ---
app.layout = html.Div([
    html.H1("📈 Hygdra Forecasting - Trader Dashboard"),
    html.P("Sélectionnez un ticker et cliquez sur 'Charger les données'."),

    html.Div([
        dcc.Dropdown(
            id="ticker-dropdown",
            options=[{"label": t, "value": t} for t in load_tickers()],
            value="BTC-USD",
            placeholder="Sélectionnez un ticker",
            style={"width": "40%"}
        ),
        html.Button("Charger les données", id="load-data-btn", n_clicks=0),
    ], style={"display": "flex", "gap": "10px"}),

    html.Div(id="error-message", style={"color": "red", "margin-top": "10px"}),

    # Chart 2: Predictions vs Réel
    html.H2("📈 Prédictions vs Réel"),
    dcc.Loading(
        id="loading-pred",
        type="circle",
        children=dcc.Graph(id="pred-chart")
    ),
])

# --- Callback to Update Data and Charts ---
@app.callback(
    [Output("pred-chart", "figure"),
     Output("error-message", "children")],
    Input("load-data-btn", "n_clicks"),
    State("ticker-dropdown", "value")
)
def update_data(n_clicks, selected_ticker):
    if not selected_ticker:
        return dash.no_update, dash.no_update, "Veuillez sélectionner un ticker."

    df = load_stock_data(selected_ticker)

    if df.empty:
        return dash.no_update, dash.no_update, "Aucune donnée disponible."

    # Convert date columns to datetime objects (if available)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if "pred_date" in df.columns:
        df["pred_date"] = pd.to_datetime(df["pred_date"])

    df.sort_values("Date", inplace=True)

    # --- Chart 2: Predictions vs Réel ---
    # We use graph_objects so that each trace can have its own x values.
    fig_pred = go.Figure()

    # Trace for the actual closing prices (using "Date")
    fig_pred.add_trace(go.Scatter(
        x=df["Date"],
        y=df[f"{selected_ticker}_close"],
        mode="lines+markers",
        name="Close"
    ))

    # Trace for the predicted prices (using "pred_date")
    if f"{selected_ticker}_pred" in df.columns and "pred_date" in df.columns:
        fig_pred.add_trace(go.Scatter(
            x=df["pred_date"],
            y=df[f"{selected_ticker}_pred"],
            mode="lines+markers",
            name="Prediction"
        ))

    fig_pred.update_layout(
        title=f"Prédictions vs Réel - {selected_ticker}",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    return fig_pred, ""

# --- Run the Application ---
if __name__ == "__main__":
    app.run_server(debug=True, port=8000)
