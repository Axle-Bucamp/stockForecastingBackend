from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Optional, List
from datamodel.sequence import SequenceRequest
from datamodel.ticker_cluster import TKGroup, TKGroupName
import redis
from os import getenv
from io import StringIO

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
    
groups_name = 'big'
print(load_data_for_group(f"{groups_name}"))


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

groups_name = 'BTC-USD'
resp = get_data_by_stock(f"{groups_name}")
json_str = StringIO(resp.json().decode('utf-8'))
df = pd.read_json(json_str, orient="records")
print(pd.DataFrame(df))