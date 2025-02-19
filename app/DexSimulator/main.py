import time
from oneinch_py import OneInchSwap, TransactionHelper, OneInchOracle
import configparser


config = configparser.ConfigParser()
config.read('.key')

# Set up configuration (use environmental variables for private key and RPC URLs in real applications)
rpc_url = "https://bsc-pokt.nodies.app"
binance_rpc = "https://binance.llamarpc.com"
# json list extracted from rpc endpoint ?

public_key = config["PUBLIC_KEY"]
private_key = config["PRIVATE_KEY"]
api_key = config["API_key"]
# 1 Inch API key

# Initialize objects for Ethereum and Binance chain
exchange = OneInchSwap(api_key, public_key)
bsc_exchange = OneInchSwap(api_key, public_key, chain='binance')
helper = TransactionHelper(api_key, rpc_url, public_key, private_key)
bsc_helper = TransactionHelper(api_key, binance_rpc, public_key, private_key, chain='binance')
oracle = OneInchOracle(rpc_url, chain='ethereum')



# See chains currently supported by the helper method:
helper.chains
# {"ethereum": "1", "binance": "56", "polygon": "137", "avalanche": "43114"}

# Straight to business:
# Get a swap and do the swap
result = exchange.get_swap("USDT", "ETH", 10, 0.5) # get the swap transaction
result = helper.build_tx(result) # prepare the transaction for signing, gas price defaults to fast.
result = helper.sign_tx(result) # sign the transaction using your private key
result = helper.broadcast_tx(result) #broadcast the transaction to the network and wait for the receipt. 

## If you already have token addresses you can pass those in instead of token names to all OneInchSwap functions that require a token argument
result = exchange.get_swap("0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9", "0x43dfc4159d86f3a37a5a4b3d4580b888ad7d4ddd", 10, 0.5) 


#USDT to ETH price on the Oracle. Note that you need to indicate the token decimal if it is anything other than 18.
oracle.get_rate_to_ETH("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", src_token_decimal=6)

# Get the rate between any two tokens.
oracle.get_rate(src_token="0x6B175474E89094C44Da98b954EedeAC495271d0F", dst_token="0x111111111117dC0aa78b770fA6A738034120C302")

exchange.health_check()
# 'OK'

# Address of the 1inch router that must be trusted to spend funds for the swap
exchange.get_spender()

# Generate data for calling the contract in order to allow the 1inch router to spend funds. Token symbol or address is required. If optional "amount" variable is not supplied (in ether), unlimited allowance is granted.
exchange.get_approve("USDT")
exchange.get_approve("0xdAC17F958D2ee523a2206206994597C13D831ec7", amount=100)

# Get the number of tokens (in Wei) that the router is allowed to spend. Option "send address" variable. If not supplied uses address supplied when Initialization the exchange object. 
exchange.get_allowance("USDT")
exchange.get_allowance("0xdAC17F958D2ee523a2206206994597C13D831ec7", send_address="0x12345")

# Token List is stored in memory
exchange.tokens
# {
#  '1INCH': {'address': '0x111111111117dc0aa78b770fa6a738034120c302',
#            'decimals': 18,
#            'logoURI': 'https://tokens.1inch.exchange/0x111111111117dc0aa78b770fa6a738034120c302.png',
#            'name': '1INCH Token',
#            'symbol': '1INCH'},
#   'ETH': {'address': '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
#          'decimals': 18,
#          'logoURI': 'https://tokens.1inch.exchange/0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png',
#          'name': 'Ethereum',
#          'symbol': 'ETH'},
#   ......
# }

# Returns the exchange rate of two tokens. 
# Tokens can be provided as symbols or addresses
# "amount" is supplied in ether
# NOTE: When using custom tokens, the token decimal is assumed to be 18. If your custom token has a different decimal - please manually pass it to the function (decimal=x)
# Also returns the "price" of more expensive token in the cheaper tokens. Optional variables can be supplied as **kwargs
exchange.get_quote(from_token_symbol='ETH', to_token_symbol='USDT', amount=1)
# (
#     {
#         "fromToken": {
#             "symbol": "ETH",
#             "name": "Ethereum",
#             "decimals": 18,
#             "address": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
#             "logoURI": "https://tokens.1inch.io/0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png",
#             "tags": ["native"],
#         },
#         "toToken": {
#             "symbol": "USDT",
#             "name": "Tether USD",
#             "address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
#             "decimals": 6,
#             "logoURI": "https://tokens.1inch.io/0xdac17f958d2ee523a2206206994597c13d831ec7.png",
#             "tags": ["tokens"],
#         ...
#     Decimal("1076.503093"),
# )

# Creates the swap data for two tokens.
# Tokens can be provided as symbols or addresses
# Optional variables can be supplied as **kwargs
# NOTE: When using custom tokens, the token decimal is assumed to be 18. If your custom token has a different decimal - please manually pass it to the function (decimal=x)

exchange.get_swap(from_token_symbol='ETH', to_token_symbol='USDT', amount=1, slippage=0.5)
# {
#     "fromToken": {
#         "symbol": "ETH",
#         "name": "Ethereum",
#         "decimals": 18,
#         "address": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
#         "logoURI": "https://tokens.1inch.io/0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png",
#         "tags": ["native"],
#     },
#     "toToken": {
#         "symbol": "USDT",
#         "name": "Tether USD",
#         "address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
#         "decimals": 6,
#         "logoURI": "https://tokens.1inch.io/0xdac17f958d2ee523a2206206994597c13d831ec7.png",
#         "tags": ["tokens"],
#
#     ...
#
#     ],
#     "tx": {
#         "from": "0x1d05aD0366ad6dc0a284C5fbda46cd555Fb4da27",
#         "to": "0x1111111254fb6c44bac0bed2854e76f90643097d",
#         "data": "0xe449022e00000000000000000000000000000000000000000000000006f05b59d3b20000000000000000000000000000000000000000000000000000000000001fed825a0000000000000000000000000000000000000000000000000000000000000060000000000000000000000000000000000000000000000000000000000000000140000000000000000000000011b815efb8f581194ae79006d24e0d814b7697f6cfee7c08",
#         "value": "500000000000000000",
#         "gas": 178993,
#         "gasPrice": "14183370651",
#     },
# }