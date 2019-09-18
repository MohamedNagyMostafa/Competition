import pandas
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.client import device_lib
import pandas as pd
columns_new = ['uniqueid', 'country', 'year', 'location_type',
       'cellphone_access', 'household_size', 'age_of_respondent',
       'gender_of_respondent', 'relationship_with_head', 'marital_status',
       'education_level', 'job_type', 'bank_account']
columns_new_t = ['uniqueid', 'country', 'year', 'location_type',
       'cellphone_access', 'household_size', 'age_of_respondent',
       'gender_of_respondent', 'relationship_with_head', 'marital_status',
       'education_level', 'job_type']

train_dataset = pd.read_csv('data/Train_v2.csv')
test_dataset = pd.read_csv('data/Test_v2.csv')

test_country = test_dataset['country']
test_id = test_dataset['uniqueid']

train_dataset = train_dataset[columns_new]
test_dataset = test_dataset[columns_new_t]

X = train_dataset.iloc[:,:-1].values
y = train_dataset.iloc[:, -1].values
X_t = test_dataset.to_numpy()





def encoding(X):
	labelencoder_X1 = LabelEncoder()
	X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
	labelencoder_X2 = LabelEncoder()
	X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
	labelencoder_X3 = LabelEncoder()
	X[:, 3] = labelencoder_X3.fit_transform(X[:, 3])
	labelencoder_X4 = LabelEncoder()
	X[:, 4] = labelencoder_X4.fit_transform(X[:, 4])
	labelencoder_X7 = LabelEncoder()
	X[:, 7] = labelencoder_X7.fit_transform(X[:, 7])
	labelencoder_X8 = LabelEncoder()
	X[:, 8] = labelencoder_X8.fit_transform(X[:, 8])
	labelencoder_X9 = LabelEncoder()
	X[:, 9] = labelencoder_X9.fit_transform(X[:, 9])
	labelencoder_X10 = LabelEncoder()
	X[:, 10] = labelencoder_X10.fit_transform(X[:, 10])
	labelencoder_X11 = LabelEncoder()
	X[:, 11] = labelencoder_X11.fit_transform(X[:, 11])
	
		
	return X

def hot_encoding(X):
	onehotencoder1 = OneHotEncoder(categorical_features = [0,1,7,8,9,10])
	X = onehotencoder1.fit_transform(X).toarray()
	return X

X = encoding(X)
X = X[:,1:]
X = hot_encoding(X)

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dropout(0.3))


# Adding the second hidden layer
classifier.add(Dense(output_dim = 256, activation = 'relu'))

classifier.add(Dropout(0.3))

# Adding the output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 8, nb_epoch = 20)

X_t = encoding(X_t)
X_t = X_t[:,1:]
X_t = hot_encoding(X_t)
out = classifier.predict_classes(X_t).squeeze()

test_dataset_id = test_id + ' x ' + test_country

from pandas import DataFrame

Cars = {'unique_id': test_dataset_id,
        'bank_account':out 
        }

df = DataFrame(Cars, columns= ['unique_id', 'bank_account'])

export_csv = df.to_csv ('output/finish.csv', index = None, header=True)
