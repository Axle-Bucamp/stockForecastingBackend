from pydantic import BaseModel
from hygdraDexSim.dataclass.action import Action


# FastAPI Models
class Transaction(BaseModel):
    token_from: str
    token_to: str
    amount: float = 0
    action: Action