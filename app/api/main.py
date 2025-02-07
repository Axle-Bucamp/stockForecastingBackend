from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Optional, List
from datamodel.sequence import SequenceRequest
from datamodel.ticker_cluster import TKGroup, TKGroupName
import redis
from os import getenv
from io import StringIO

# FastAPI App Initialization
app = FastAPI()

# Initialize Redis client (adjust host, port, db as needed)
redis_client = redis.Redis(
    host=getenv("REDIS_HOST", "localhost"), 
    port=int(getenv("REDIS_PORT", "6379")),  # Ensure port is an integer
    db=0
)

def load_data_for_group(group_name: str) -> pd.DataFrame:
    """
    Load stock data for a given group from Redis.
    
    Args:
        group_name (str): The group name (e.g., "tech").
    
    Returns:
        pd.DataFrame: DataFrame containing the stock data.
    
    Raises:
        HTTPException: If reading the data from Redis fails.
    """
    try:
        key = f"{group_name}_days"
        json_data = redis_client.get(key)
        if not json_data:
            raise HTTPException(status_code=404, detail=f"No data found in Redis for group '{group_name}'")
        # Decode the JSON bytes to string and convert to DataFrame.
        json_str = StringIO(json_data.decode('utf-8'))
        df = pd.read_json(json_str, orient="records")
        # Ensure the date column is parsed correctly (assumes column name "Date")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load data for group '{group_name}' from Redis: {str(e)}"
        )

def load_data_for_group_from_file(group_name: str) -> pd.DataFrame:
    """
    Load stock data for a given group from a CSV file.
    
    Args:
        group_name (str): The group name (e.g., "tech").
    
    Returns:
        pd.DataFrame: DataFrame containing the stock data.
    
    Raises:
        HTTPException: If reading the CSV file fails.
    """
    try:
        df = pd.read_csv(f'data/{group_name}_days.csv', parse_dates=['Date'])
        return df
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load data for group '{group_name}': {str(e)}"
        )

def get_group_for_ticker(ticker: str) -> Optional[str]:
    """
    Return the group name for a given ticker, or None if not found.
    
    Args:
        ticker (str): The stock ticker symbol.
    
    Returns:
        Optional[str]: The group name if found, else None.
    """
    for group in TKGroup.__members__.values():
        group_name, tickers = group.value
        if ticker in tickers:
            return group_name
    return None

# -----------------------------
# Attribute Endpoints
# -----------------------------

@app.get("/attribute/ticker/list", response_model=List[str])
async def get_ticker_list():
    """
    Get a list of all tickers from the defined ticker groups.
    
    Returns:
        List[str]: List of all available ticker symbols.
    """
    all_tickers = []
    for group in TKGroup.__members__.values():
        # group.value[1] is the list of tickers for that group
        all_tickers.extend(group.value[1])
    return all_tickers

@app.get("/attribute/ticker-cluster/name")
async def get_ticker_cluster_names():
    """
    Get the names of the ticker clusters.
    
    Returns:
        List[str]: List of cluster names defined in TKGroupName.
    """
    return list(TKGroupName.__members__.keys())

@app.get("/attribute/ticker-cluster/list")
async def get_ticker_cluster_list():
    """
    Get the full ticker cluster mapping.
    
    Returns:
        dict: A dictionary mapping cluster names to their corresponding values.
    """
    return {name: member.value for name, member in TKGroup.__members__.items()}

@app.get("/attribute/seq")
async def get_sequence_requests():
    """
    Get the available sequence requests.
    
    Returns:
        List[str]: List of sequence request names defined in SequenceRequest.
    """
    return list(SequenceRequest.__members__.keys())

# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.get("/predict/{ticker}/{seq}")
def predict_stock(ticker: str, seq: SequenceRequest):
    """
    Predict stock data for a given ticker and sequence request.
    
    Currently, this endpoint returns the raw CSV data for the corresponding group.
    
    Args:
        ticker (str): The stock ticker symbol.
        seq (SequenceRequest): The sequence request (not used in the current logic).
    
    Returns:
        JSONResponse: Raw CSV data in JSON format.
    
    Raises:
        HTTPException: If the ticker is not found in any group or if the CSV file cannot be read.
    """
    group_name = get_group_for_ticker(ticker)
    if not group_name:
        raise HTTPException(status_code=400, detail="Ticker not found in groups")

    try:
        df = pd.read_csv(f'data/{group_name}_days.csv', parse_dates=['Date'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")
    
    return JSONResponse(content=df.to_json(orient="records", date_format="iso"))

# -----------------------------
# Data Retrieval Endpoints
# -----------------------------

@app.get("/api/json")
def get_data():
    """
    Fetch all stock data as JSON.
    
    Returns:
        JSONResponse: Stock data in JSON format.
    """
    # For demonstration, use a default group (e.g., "tech"). Adjust as needed.
    default_group = "tech"
    df = load_data_for_group(default_group)
    
    return JSONResponse(content=df.to_json(orient="records", date_format="iso"))

@app.get("/api/json/stock/{ticker}")
def get_data_by_stock(ticker: str):
    """
    Fetch the data for a specific stock using its ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol.
    
    Returns:
        JSONResponse: Stock data for the ticker in JSON format.
    
    Raises:
        HTTPException: If the ticker is not found in any group or if the stock data is missing.
    """
    group_name = get_group_for_ticker(ticker)
    if not group_name:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not found in groups.")
    
    df = load_data_for_group(group_name)
    if f"{ticker}_close" not in df.columns:
        raise HTTPException(status_code=404, detail=f"Stock data for {ticker} not found.")
    
    result = df[[f"{ticker}_close", f"{ticker}_pred", "Date"]]
    return JSONResponse(content=result.to_json(orient="records", date_format="iso"))

@app.get("/api/json/stockatdate/{ticker}/{date}")
def get_stock_at_date(ticker: str, date: str):
    """
    Fetch stock data for a specific stock on a given date.
    
    Args:
        ticker (str): The stock ticker symbol.
        date (str): The date (format: yyyy-mm-dd).
    
    Returns:
        JSONResponse: Stock data for the ticker on the given date in JSON format.
    
    Raises:
        HTTPException: If the ticker is not found, if the date format is invalid, or if no data is found.
    """
    group_name = get_group_for_ticker(ticker)
    if not group_name:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not found in groups.")
    
    try:
        timestamp = pd.Timestamp(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format: use yyyy-mm-dd.")
    
    df = load_data_for_group(group_name)
    if f"{ticker}_close" not in df.columns or f"{ticker}_pred" not in df.columns:
        raise HTTPException(status_code=404, detail=f"Stock data for {ticker} not found.")
    
    result = df[df["Date"] == timestamp][[f"{ticker}_close", f"{ticker}_pred", "Date"]]
    if result.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified date.")
    
    return JSONResponse(content=result.to_json(orient="records", date_format="iso"))

@app.get("/api/json/stockindaterange/{ticker}/{date_begin}/{date_end}")
def get_stock_in_daterange(ticker: str, date_begin: str, date_end: str):
    """
    Fetch stock data for a specific stock within a given date range.
    
    Args:
        ticker (str): The stock ticker symbol.
        date_begin (str): Start date (format: yyyy-mm-dd).
        date_end (str): End date (format: yyyy-mm-dd).
    
    Returns:
        JSONResponse: Stock data within the specified date range in JSON format.
    
    Raises:
        HTTPException: If the ticker is not found, if date formats are invalid, or if no data is found.
    """
    group_name = get_group_for_ticker(ticker)
    if not group_name:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not found in groups.")
    
    try:
        start_date = pd.Timestamp(date_begin)
        end_date = pd.Timestamp(date_end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format: use yyyy-mm-dd.")
    
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date must be earlier than or equal to end date.")
    
    df = load_data_for_group(group_name)
    if f"{ticker}_close" not in df.columns or f"{ticker}_pred" not in df.columns:
        raise HTTPException(status_code=404, detail=f"Stock data for {ticker} not found.")
    
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    result = df.loc[mask][[f"{ticker}_close", f"{ticker}_pred", "Date"]]
    return JSONResponse(content=result.to_json(orient="records", date_format="iso"))
