import requests

method = "get"
apiUrl = "https://api.1inch.dev/swap/v6.0/1/quote"
requestOptions = {
      "headers": {
  "Authorization": "Bearer wwBHIxQdQ0MeDKvYt3thRtZ1LXzsJFPm"
},
      "body": "",
      "params": {}
}

# Prepare request components
headers = requestOptions.get("headers", {})
body = requestOptions.get("body", {})
params = requestOptions.get("params", {})


response = requests.get(apiUrl, headers=headers, params=params)

print(response.json())