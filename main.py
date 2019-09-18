from data_handling import *

from util import *
from processes import *
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


def change_lr(g,value):
	for l in g.param_groups:
		l['lr'] = value

batch_size = 12

train_dataset, train_labels, test_dataset, train_dataset_id, test_dataset_id, country_set = read_data('data/Train_v2.csv', 'data/Test_v2.csv')
(train_dataset, train_labels, train_id) = shuffleAndSplit(train_dataset, train_labels, train_dataset_id, ratio=None)
#Saving
#print(train_dataset.shape)
#saver([train_dataset, train_labels, train_id, val_dataset, val_labels, val_id, test_dataset, test_dataset_id, country_set])
#train_dataset, train_labels, train_id, val_dataset, val_labels, val_id, test_dataset, test_dataset_id, country_set = loader(9)
#print(train_dataset.shape)
'''
end =  int(len(val_dataset) * 0.33)
split_dataset, val_dataset, split_labels,val_labels  = val_dataset[:end], val_dataset[end:], val_labels[:end], val_labels[end:]
train_dataset = np.append(train_dataset,split_dataset, axis=0)
train_labels = np.append(train_labels,split_labels, axis=0)

'''
#svm(train_dataset, train_labels, val_dataset, val_labels)
trainLoader, testLoader = to_DataLoader(train_dataset, train_labels, batch_size=batch_size), to_DataLoader(test_dataset, batch_size=len(test_dataset))
#87.94617563739376
'''
model = nn.Sequential(OrderedDict([
	('fc1', nn.Linear(len(train_dataset[0]), 105)),
	('relu1', nn.ReLU()),
	('dropout1', nn.Dropout(0.3)),
	('fc2', nn.Linear(105,64)),
	('relu2', nn.ReLU()),
	('dropout2', nn.Dropout(0.3)),
	('fc3', nn.Linear(64,2))
	]))
'''
'''
model = nn.Sequential(OrderedDict([
	('fc1', nn.Linear(len(train_dataset[0]), 100)),
	('relu1', nn.ReLU ()),
	('dropout1', nn.Dropout(0.5)),
	('fc2', nn.Linear(100,100)),
	('relu8', nn.ReLU ()),
	('dropout2', nn.Dropout(0.5)),
	('fc8', nn.Linear(100,2)),
	]))
lr = 0.1

optimizer = optim.Adadelta(model.parameters(), lr=lr)
#check_point = torch.load('save/new11.pt')
#model = model.cuda()
#model.load_state_dict(check_point['state_dict'])

#optimizer.load_state_dict(check_point['optimizer'])

'''
#model = model.cuda()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.AdamTanh (model.parameters(), lr=0.0001)
'''


criterion = nn.CrossEntropyLoss()

test_loss =np.inf
	#(train_dataset, train_labels, train_id) = shuffleAndSplit(train_dataset, train_labels, train_id, ratio=None)
test_loss = train(test_loss = test_loss, model=model, epochs=1000, criterion=criterion, optimizer=optimizer, trainLoader=trainLoader, testLoader=valLoader, batch_size=batch_size)

'''
'''
check_point =  torch.load('save/new9.pt')
model.load_state_dict(check_point['state_dict'])
for data in testLoader:
	out = model(data[0])

_, pred = torch.max(out, 1)
test_dataset = pd.read_csv('data/Test_v2.csv')
country_set = test_dataset['country']
correct_tensor = pred.numpy()
'''
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
