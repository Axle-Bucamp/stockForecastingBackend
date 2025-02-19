from pydantic import BaseModel
from hygdraDexSim.dataclass.chain import Chain


# FastAPI Models
class Wallet(BaseModel):
    chain: Chain
    address: str
    pk : str
    amount: float = 0
