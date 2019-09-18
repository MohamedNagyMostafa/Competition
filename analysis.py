import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_dataset = pd.read_csv('data/Train_v2.csv')
test_dataset = pd.read_csv( 'data/Test_v2.csv')

def compare(data1, data2):
	return np.logical_and(data1.to_numpy(), data2.to_numpy())

def compare_sum(data1, data2):
	return np.sum(compare(data1, data2))

Rural_Account = compare_sum(train_dataset['bank_account'] == 'Yes',train_dataset['location_type'] == 'Rural')
Urban_Account = compare_sum(train_dataset['bank_account'] == 'Yes',train_dataset['location_type'] == 'Urban')
household_size_Account = compare_sum(train_dataset['bank_account'] == 'Yes',train_dataset['household_size'] > 15)
Cell_Account = compare_sum(train_dataset['bank_account'] == 'Yes',train_dataset['cellphone_access'] == 'Yes')

print(train_dataset.loc[3])
print('Have accounts : {}, Have not accounts: {}, location type: {} - {} - have account respectively: {}, {} \n household_size > 5 : {}, cell access: {}'.format(
	sum([train_dataset['bank_account'] == 'Yes'][0]), 
	sum([train_dataset['bank_account'] == 'No'][0]),
	len(set(train_dataset['location_type'])),
	set(train_dataset['location_type']),
	Rural_Account,Urban_Account, household_size_Account, Cell_Account
	)
)

print(np.min(train_dataset['household_size']))

plt.scatter(train_dataset['household_size'].to_numpy(), [train_dataset['bank_account'] == 'Yes'][0].to_numpy().astype(int))
plt.show()

plt.scatter(train_dataset['age_of_respondent'].to_numpy(), [train_dataset['bank_account'] == 'Yes'][0].to_numpy().astype(int))
plt.show()

plt.scatter(train_dataset['year'].to_numpy(), [train_dataset['bank_account'] == 'Yes'][0].to_numpy().astype(int))
plt.show()
