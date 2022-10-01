import requests
url = 'http://192.168.1.7:8080/sang-al'
todo = {"userID": 1, "emb": 3}
reponse = requests.post(url=url, json=todo) 
dict_json = reponse.json()
print(dict_json)