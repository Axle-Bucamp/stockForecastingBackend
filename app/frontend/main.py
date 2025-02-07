import dash
from dash import dcc, html, Input, Output, State, ctx
import requests
import pandas as pd
import plotly.express as px
import os
from io import StringIO


# --- Configuration API ---
host = os.getenv("API_HOST", "localhost")
port = os.getenv("API_PORT", "7060")
API_BASE_URL = f"http://{host}:{port}"

# --- Initialisation de l'application Dash ---
app = dash.Dash(__name__)
server = app.server

# --- Chargement initial des tickers ---
def load_tickers():
    try:
        resp = requests.get(f"{API_BASE_URL}/attribute/ticker/list")
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"Erreur de chargement des tickers: {e}")
        return []

# --- Fonction pour charger les données du stock ---
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

# --- Interface Utilisateur Dash ---
app.layout = html.Div([
    html.H1("📈 Hygdra Forecasting - Trader Dashboard"),
    html.P("Sélectionnez un ticker et cliquez sur 'Charger les données'."),

    html.Div([
        dcc.Dropdown(
            id="ticker-dropdown",
            options=[{"label": t, "value": t} for t in load_tickers()],
            value=None,
            placeholder="Sélectionnez un ticker",
            style={"width": "40%"}
        ),
        html.Button("Charger les données", id="load-data-btn", n_clicks=0),
    ], style={"display": "flex", "gap": "10px"}),

    html.Div(id="error-message", style={"color": "red", "margin-top": "10px"}),

    html.H2("📊 Données de Stock"),
    dcc.Loading(
        id="loading-table",
        type="circle",
        children=dcc.Graph(id="stock-table")
    ),

    html.H2("📈 Graphique des Prix"),
    dcc.Loading(
        id="loading-chart",
        type="circle",
        children=dcc.Graph(id="stock-chart")
    ),
])

# --- Callback pour Charger les Données ---
@app.callback(
    [Output("stock-table", "figure"),
     Output("stock-chart", "figure"),
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

    # Transformation des données
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime('%Y-%m-%d')
        df.sort_values("Date", inplace=True)

    # Création du Tableau
    fig_table = px.line(df, x="Date", y=f"{selected_ticker}_close",
                        title=f"Prix de clôture pour {selected_ticker}",
                        labels={"Date": "Date", f"{selected_ticker}_close": "Prix de Clôture"})

    # Création du Graphique
    fig_chart = px.line(df, x="Date",
                        y=[f"{selected_ticker}_close", f"{selected_ticker}_pred"],
                        title=f"Prédictions vs Réel - {selected_ticker}",
                        labels={"Date": "Date"},
                        markers=True)

    return fig_table, fig_chart, ""

# --- Lancement de l'application ---
if __name__ == "__main__":
    app.run_server(debug=True, port=8000)

