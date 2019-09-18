import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
def train(test_loss, model, criterion, optimizer, trainLoader, testLoader,batch_size,epochs=10):
	cuda = torch.cuda.is_available()

	if cuda: model = model.cuda()
	counter = 0
	for epoch in range(epochs):
		train_cost = 0.0
		test_cost = 0.0

		model.train()
		for data, label in trainLoader:
			
			if cuda: data, label = data.cuda(), label.cuda()

			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, label)

			loss.backward()
			optimizer.step()

			train_cost += loss.item()*len(data)
		else:
			model.eval()
			total = 0.0
			correct = 0.0
			for data, label in testLoader:
				if cuda: data, label = data.cuda(), label.cuda()

				output = model(data)
				loss = criterion(output, label)
				test_cost += loss.item()*len(data)

				_, pred = torch.max(output, 1)

				correct_tensor = pred.eq(label.data.view_as(pred))
				correct += np.sum(correct_tensor.numpy() if not cuda else correct_tensor.cpu().numpy())
				
			
			acc = correct * 100 / (len(testLoader) * batch_size)
			loss_test_c = test_cost/len(testLoader)
			print('epoch: {} --> train loss = {}, test loss = {}, accuracy = {}'.format(epoch, train_cost/len(trainLoader), test_cost/len(testLoader), acc))
			counter += 1
			if test_loss > loss_test_c:
				test_loss = loss_test_c
				counter = 0
				print('save')
				check_point = {
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict()}
				torch.save(check_point, 'save/new12.pt')
			elif counter == 30:
				break
	return test_loss

def svm(data_train, label_train, data_test, label_test= None):

	svclassifier = SVC(kernel='rbf',degree =6)
	
	svclassifier.fit(data_train, label_train)
	
	return svclassifier.predict(data_test)



