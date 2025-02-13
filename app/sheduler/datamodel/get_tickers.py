import requests
import json

# Fetch data from Kraken API
url = 'https://api.kraken.com/0/public/Assets'
response = requests.get(url)
data = response.json()

# Extract 'altname' for enabled assets
enabled_assets = [
    asset_info['altname']
    for asset_info in data['result'].values()
    if asset_info.get('status') == 'enabled'
]

# Save the list to a JSON file
with open('enabled_assets.json', 'w') as file:
    json.dump(enabled_assets, file, indent=4)

print(f"Extracted {len(enabled_assets)} enabled assets.")
