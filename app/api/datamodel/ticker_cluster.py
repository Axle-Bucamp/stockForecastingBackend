from enum import Enum

# trier
# --- use later ---
class TKGroup(Enum):
    crypto10 = ("crypto-standard", ["ADA-USD", "SOL-USD", 'XRP-USD', "ETH-USD", "BTC-USD", 'TON-USD'])
    freedom = ("thisiselonmusk", ["XRP-USD", "SHIB-USD", 'DOGE-USD', 'TRUMP-USD', 'SOL-USD', 'BTC-USD'])
    cac40 = ("cac40", ["AMD", "INTC", 'GOOG', "NVDA", "META", "GOOGL", "AAPL", "MSFT"])
    big = ("big", ["SQ", "BA", "PFE", "PYPL", "COST", "SBUX", "DIS", "NFLX", "JNJ"])
    TECH = ("tech", ["AAPL", "MSFT", "GOOGL"])
    FINANCE = ("finance", ["JPM", "GS", "BAC"])
    ENERGY = ("energy", ["XOM", "CVX", "BP"])
'''  
    memecoin = ("memecoin", ['BONK-USD', 'NOT-USD', 'SHIB-USD', 'PEPECOIN-USD', 'DOGE-USD'])
    defi = ("defi", ['AVAX-USD', 'Link-USD', 'HBAR-USD', 'UNI-USD', 'STX-USD', 'AVEE-USD', 'OM-USD', 'FTM-USD', "INJ-USD"])
    web3 = ("web3", ['Link-USD', 'ICP-USD', 'FET-USD', 'FIL-USD', "CRO-EUR", 'THETA-USD', 'GRT-USD'])
''' 

# --- use later ---
class TKGroupName(str, Enum):
    crypto10 = "crypto-standard"
    memecoin = "memecoin"
    freedom = "thisiselonmusk"
    defi = "defi"
    web3 = "web3"
    cac40 = "cac40"
    big = "big"
