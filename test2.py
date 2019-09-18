from data_handling import *
from keras.utils import to_categorical
import tensorflow as tf
from util import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

train_dataset, train_labels, test_dataset, train_dataset_id, test_dataset_id, country_set = read_data('data/Train_v2.csv', 'data/Test_v2.csv')
(train_dataset, train_labels, train_id) , (val_dataset, val_labels, val_id) = shuffleAndSplit(train_dataset, train_labels, train_dataset_id, ratio=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(train_dataset)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(X, train_labels)


'''
model = Sequential([
  Dense(200, activation='relu', input_shape=(len(train_dataset[0]), )),
  Dropout(0.3),
  Dense(200, activation='relu'),
  Dropout(0.3),
  Dense(200, activation='relu'),
  Dropout(0.3),
  Dense(200, activation='relu'),
  Dropout(0.3),
  Dense(2, activation='sigmoid'),
])

print(model)
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=['sparse_categorical_accuracy'])
model.fit(
  train_dataset, # training data
  train_labels, # training targets
  epochs=50,
  batch_size=len(train_dataset[0]),
  validation_data=(val_dataset, val_labels)
)

out = model.predict_classes(test_dataset)
test_dataset = pd.read_csv('data/Test_v2.csv')
country_set = test_dataset['country']

test_dataset_id = test_dataset_id + ' x ' + country_set

from pandas import DataFrame

Cars = {'unique_id': test_dataset_id,
        'bank_account':out 
        }

df = DataFrame(Cars, columns= ['unique_id', 'bank_account'])

export_csv = df.to_csv ('output/finish.csv', index = None, header=True)
'''