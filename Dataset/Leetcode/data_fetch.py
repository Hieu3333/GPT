import requests
import json

url = "https://datasets-server.huggingface.co/rows?dataset=RayBernard%2Fleetcode&config=default&split=train&offset=0&length=100"
response = requests.get(url)

if response.status_code == 200:
    data = response.json() #parse the JSON
    with open('data.json','w') as file:
        json.dump(data,file,indent=4)
    print('Download dataset successfully')
else:
    print('Error downloading dataset')
