import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg'}

result = requests.post(url, json=data).json()
print(result)