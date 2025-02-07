import requests
import os
import pandas as pd
from io import StringIO


# --- Configuration API ---
host = os.getenv("API_HOST", "localhost")
port = os.getenv("API_PORT", "7060")
API_BASE_URL = f"http://{host}:{port}"

def load_tickers():
    try:
        resp = requests.get(f"{API_BASE_URL}/attribute/ticker/list")
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"Erreur de chargement des tickers: {e}")
        return []
    
print(load_tickers())

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

print(load_stock_data("BTC-USD"))
