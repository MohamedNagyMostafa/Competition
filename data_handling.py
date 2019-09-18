import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
def read_data(dir_train, dir_test):
	
	collection = [
	'country','year','location_type', 'marital_status', 'education_level', 'job_type', 'relationship_with_head']
	non_collection = ['bank_account', 'cellphone_access','gender_of_respondent']
	numerical = ['household_size', 'age_of_respondent']

	train_dataset = pd.read_csv(dir_train)
	test_dataset = pd.read_csv(dir_test)

	encodingOneZero(set_=set(train_dataset['bank_account']), data1=train_dataset, data2=test_dataset, title='bank_account')

	country_set = test_dataset['country']

	train_dataset_id, test_dataset_id = train_dataset['uniqueid'].to_numpy(), test_dataset['uniqueid'].to_numpy()
	train_labels = train_dataset['bank_account']

	train_dataset, test_dataset = train_dataset.drop(columns=['uniqueid','relationship_with_head','location_type','gender_of_respondent','age_of_respondent','year','household_size','marital_status','bank_account']), test_dataset.drop(columns=['uniqueid','relationship_with_head','location_type','gender_of_respondent','age_of_respondent','year','household_size','marital_status'])
	#encoding(set_=set(train_dataset['uniqueid']), data1=train_dataset, data2=test_dataset, title='uniqueid')
	train_dataset, test_dataset = pre_processing(
		train_dataset,
		test_dataset,
		collection=['country', 'education_level', 'job_type'],
		non_collection=['cellphone_access'],
		numerical=[])

	
	return train_dataset.to_numpy(), train_labels.to_numpy().astype(int), test_dataset.to_numpy(), train_dataset_id, test_dataset_id, country_set

def pre_processing(train_dt, test_dt, collection, non_collection, numerical):
	
	for cell in non_collection: train_dt, test_dt = encodingOneZero(set_=set(train_dt[cell]), data1=train_dt, data2=test_dt, title=cell) 
	for cell in numerical: train_dt, test_dt = encodingNumerical(train_dt, test_dt, cell)
	for cell in collection: train_dt, test_dt = one_hot_encoding(train_dt, test_dt, cell)

	return train_dt, test_dt

def encoding(set_, data1, data2, title):
	for ind, i in enumerate(set_):
		data1[title][data1[title] == i] = ind
		data2[title][data2[title] == i] = ind

def one_hot_encoding(data1 , data2, title):
	data1 = pd.concat([data1,pd.get_dummies(data1[title], prefix=title)],axis=1)
	data1.drop([title],axis=1, inplace=True)
	data2 = pd.concat([data2,pd.get_dummies(data2[title], prefix=title)],axis=1)
	data2.drop([title],axis=1, inplace=True)
	return data1, data2

def encodingOneZero(set_, data1, title, data2):
	if title != 'bank_account':
		for ind, i in enumerate(set_):
			data1[title][data1[title]== i] = ind
			data2[title][data2[title]== i] = ind
		
	elif title in data1 and title in data2:
		for ind, i in enumerate(['No', 'Yes']):
			data1[title][data1[title]== i] = ind
			data2[title][data2[title]== i] = ind

	elif title in data1:
		for ind, i in enumerate(['No', 'Yes']):
			data1[title][data1[title]== i] = ind
	return data1, data2

def encodingNumerical(data1, data2, title):
	max_= max(data1[title].max(), data2[title].max())
	data1[title], data2[title] = data1[title]/max_, data2[title]/max_
	return data1, data2
'''
train_dataset, train_labels, test_dataset, train_dataset_id, test_dataset_id = read_data('data/Train_v2.csv', 'data/Test_v2.csv')

import seaborn as sns
import matplotlib.pyplot as plt

corrmat = train_dataset.corr()
top_corr_features = corrmat.index
#plot heat map
g=sns.heatmap(train_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


bestfeatures = SelectKBest(score_func=chi2, k=3)
fit = bestfeatures.fit(train_dataset,train_labels)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(train_dataset.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
print(train_dataset.loc[0])


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(train_dataset,train_labels)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=train_dataset.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
'''
'''
train_dataset, train_labels, test_dataset, train_dataset_id, test_dataset_id = read_data('data/Train_v2.csv', 'data/Test_v2.csv')

from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormapimport numpy as np
h = .01
x_min, x_max = -4,4
y_min, y_max = -1.5,1.5# loading dataset

X_pca = PCA().fit_transform(train_dataset)
X_selected = X_pca[:,:2]# training classifier and evaluating on the whole plane
clf = SVC(kernel='linear')
clf.fit(X_selected,train_labels)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)# Plotting

'''