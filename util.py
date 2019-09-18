import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def shuffleAndSplit(train_dataset,train_labels, train_id, ratio=0.7):
	s = len(train_dataset)

	x = np.array(range(s))
	np.random.shuffle(x)

	train_dataset, train_labels, train_id = train_dataset[x], train_labels[x], train_id[x]
	if ratio is not None:
		train_ind = s * ratio
		return (train_dataset[:int(train_ind)],train_labels[:int(train_ind)], train_id[:int(train_ind)]), (train_dataset[int(train_ind):],train_labels[int(train_ind):], train_id[int(train_ind):])
	else:
		return train_dataset, train_labels, train_id

def to_DataLoader(x, y = None, batch_size=64, shuffle=False, num_workers=0):
	if y is not None:
		train = torch.stack([torch.Tensor(i.astype(float)) for i in x]) 
		labels = torch.Tensor(y.astype(int))
		labels = labels.type(torch.LongTensor)

		dataset = TensorDataset(train,labels)
	else:
		train = torch.stack([torch.Tensor(i.astype(float)) for i in x])
		dataset = TensorDataset(train)

	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def saver(data):
	[np.save('data'+str(i), data) for i, data in enumerate(data)]

def loader(c):
	return [np.load('data'+str(i) +'.npy') for i in range(c)]
