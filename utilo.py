import pandas
from pandas import DataFrame

def upload(dir, column1, column2):

	Cars = {'unique_id': column1,'bank_account':column2}

	df = DataFrame(Cars, columns= ['unique_id', 'bank_account'])

	df.to_csv (dir, index = None, header=True)
