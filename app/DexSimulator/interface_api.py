# curl -X 'GET' \
#  'https://api.1inch.dev/swap/v5.2/1/tokens' \
#  -H 'accept: application/json' \
#  -H 'Authorization: Bearer YOUR-API-KEY'

import requests
from web3 import Web3
import asyncio 

chainId = 56  # Chain ID for Binance Smart Chain (BSC)
web3RpcUrl = "https://bsc-dataseed.binance.org"  # URL for BSC node
headers = { "Authorization": "Bearer [YOUR_API_KEY]", "accept": "application/json" }
walletAddress = "0x...xxx"  # Your wallet address
privateKey = "0x...xxx"  # Your wallet's private key. NEVER SHARE THIS WITH ANYONE!

swapParams = {
    "src": "0x111111111117dc0aa78b770fa6a738034120c302",  # Token address of 1INCH
    "dst": "0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3",  # Token address of DAI
    "amount": "100000000000000000",  # Amount of 1INCH to swap (in wei)
    "from": walletAddress,
    "slippage": 1,  # Maximum acceptable slippage percentage for the swap (e.g., 1 for 1%)
    "disableEstimate": False,  # Set to True to disable estimation of swap details
    "allowPartialFill": False,  # Set to True to allow partial filling of the swap order
}

apiBaseUrl = f"https://api.1inch.dev/swap/v6.0/{chainId}"
web3 = Web3(web3RpcUrl)

# Construct full API request URL
def apiRequestUrl(methodName, queryParams):
    return f"{apiBaseUrl}{methodName}?{'&'.join([f'{key}={value}' for key, value in queryParams.items()])}"

# Function to check token allowance
def checkAllowance(tokenAddress, walletAddress):
    url = apiRequestUrl("/approve/allowance", {"tokenAddress": tokenAddress, "walletAddress": walletAddress})
    response = requests.get(url, headers=headers)
    data = response.json()
    return data.get("allowance")

# Sign and post a transaction, return its hash
async def signAndSendTransaction(transaction, private_key):
    signed_transaction = web3.eth.account.signTransaction(transaction, private_key)
    return await broadCastRawTransaction(signed_transaction.rawTransaction)

# Prepare approval transaction, considering gas limit
async def buildTxForApproveTradeWithRouter(token_address, amount=None):
    # Assuming you have defined apiRequestUrl() function to construct the URL
    url = apiRequestUrl("/approve/transaction", {"tokenAddress": token_address, "amount": amount} if amount else {"tokenAddress": token_address})
    response = requests.get(url, headers=headers)
    transaction = response.json()

    # Estimate gas limit
    wallet_address = "YOUR_WALLET_ADDRESS"
    gas_limit = web3.eth.estimateGas(transaction, from_address=wallet_address)

    return {**transaction, "gas": gas_limit}

allowance = checkAllowance(swapParams["src"], walletAddress)
print("Allowance: ", allowance)

transactionForSign = await buildTxForApproveTradeWithRouter(swapParams.src)

ok = input("Do you want to send a transaction to approve trade with 1inch router? (y/n): ")
if ok.lower() == "y":
    # Sign and send the transaction for approval
    approveTxHash = signAndSendTransaction(transactionForSign)
    print("Approve tx hash: ", approveTxHash)


def buildTxForSwap(swapParams):
    url = apiRequestUrl("/swap", swapParams)
    swapTransaction = requests.get(url,  headers={'Authorization': f'Bearer YOUR_API_KEY'}).json()["tx"]
    return swapTransaction

ok = input("Do you want to send a transaction to exchange with 1inch router? (y/n): ")
if ok.lower() == "y":
    # You need to implement the signAndSendTransaction function to sign and send the transaction
    # Replace the following line with your actual implementation
    swapTxHash = signAndSendTransaction(swapTransaction)
    print("Swap tx hash: ", swapTxHash)