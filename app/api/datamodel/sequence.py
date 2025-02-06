from enum import Enum

# --- use later ---
class SequenceRequest(str, Enum):
    DAILY = "days"
    MONTHLY = "months"
    LIVE = "minutes"
