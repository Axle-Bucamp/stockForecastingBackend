from pydantic import BaseModel
from enum import Enum


# FastAPI Models
class Action(str, Enum, BaseModel):
    BUY: str = "BUY"
    SELL: str = "SELL"