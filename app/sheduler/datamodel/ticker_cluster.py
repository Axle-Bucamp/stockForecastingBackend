from enum import Enum

# trier
# --- use later ---
"""
    crypto10 = ("crypto-standard", ["ADA-USD", "SOL-USD", 'XRP-USD', "ETH-USD", "BTC-USD", 'TON-USD'])
    freedom = ("thisiselonmusk", ["XRP-USD", "SHIB-USD", 'DOGE-USD', 'SOL-USD', 'BTC-USD'])
    cac40 = ("cac40", ["AMD", "INTC", 'GOOG', "NVDA", "META", "GOOGL", "AAPL", "MSFT"])
    big = ("big", ["SQ", "BA", "PFE", "PYPL", "COST", "SBUX", "DIS", "NFLX", "JNJ"])
    TECH = ("tech", ["AAPL", "MSFT", "GOOGL"])
    FINANCE = ("finance", ["JPM", "GS", "BAC"])
    ENERGY = ("energy", ["XOM", "CVX", "BP"]) 
    DEFI = ("defi", ['AVAX-USD', 'LINK-USD', 'UNI-USD', 'STX-USD', 'FTM-USD', "INJ-USD"])
    WEB3 = ("web3", ['LINK-USD', "SUI-USD", "CRO-EUR", 'THETA-USD'])
    MEMECOIN = ("memecoin", ['BONK-USD', 'SHIB-USD', 'PEPECOIN-USD', 'DOGE-USD'])
"""

class TKGroup(Enum):
    CRYPTO_STANDARD = ("crypto-standard", ["ADA-USD", "SOL-USD", "XRP-USD", "ETH-USD", "BTC-USD"])
    FREEDOM = ("thisiselonmusk", ["XRP-USD", "SHIB-USD", "SOL-USD", "BTC-USD"])
    WEB3 = ("web3", ["LINK-USD", "SOL"])
    GAME = ('gaming', ['SAND', 'IMX', "GALA", "AXS", "MANA"])
    DEFI = ("defi", ['AVAX-USD', 'LINK-USD', 'UNI-USD', 'STX-USD', 'FTM-USD', "INJ-USD"])
    MEMECOIN = ("memecoin", [ "SHIB-USD"])

# --- use later ---
class TKGroupName(str, Enum):
    crypto10 = "crypto-standard"
    memecoin = "memecoin"
    freedom = "thisiselonmusk"
    game = "gaming"
    defi = "defi"
    web3 = "web3"
    #cac40 = "cac40"
    #big = "big"
    #tech = "tech"
    #finance = "finance"
    #energie = "energie"

