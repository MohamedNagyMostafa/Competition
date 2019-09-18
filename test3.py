import pandas as pd


file = pd.read_csv('data/Train_v2.csv')

countries = set(file['country'])

print(file['country'])

for index, value in enumerate(countries):
	file['country'][file['country'] == value] = index

print(file['country'])


