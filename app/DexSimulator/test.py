import requests

resp = requests.get("https://api.1inch.dev/gas-price/v1.4/1")
print(resp)