import json

# print(dir(json))

# немного json вам в ленту
x = '{"name" : "Николай", "age" : 22, "city" : "Новочеркасск"}'

y = json.loads(x)

print(y["age"])