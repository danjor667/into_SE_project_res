import requests

url = "http://127.0.0.1:8000/predict/"

data = {"features": [11,143,94,33,146,36.6,0.254,51]}

response = requests.post(url, data)
print(response)