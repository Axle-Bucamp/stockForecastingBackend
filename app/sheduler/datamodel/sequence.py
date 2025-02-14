from enum import Enum

# --- use later ---
class SequenceRequest(str, Enum):
    DAILY = "days"
    LIVE = "minutes"
    HOURS = "hours"
    THIRTY = "thirty"