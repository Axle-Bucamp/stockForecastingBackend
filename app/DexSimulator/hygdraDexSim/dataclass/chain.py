from pydantic import BaseModel
from typing import Optional


# FastAPI Models
class Chain(BaseModel):
    name: str
    contract_address: str
    # main_net: Optional[Chain]