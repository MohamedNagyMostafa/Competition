import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt




def read_data(dir_train, dir_test):
	
	train_dataset = pd.read_csv(dir_train)
	test_dataset = pd.read_csv(dir_test)
	pre_processing(train_dataset, test_dataset)

	train_dataset_id, test_dataset_id = train_dataset['uniqueid'].to_numpy(), test_dataset['uniqueid'].to_numpy()
	train_labels = train_dataset['bank_account']

	
	train_dataset = train_dataset.drop(columns=['bank_account'])#, test_dataset.drop(columns=[])



	return train_dataset, train_labels.astype(int), test_dataset, train_dataset_id, test_dataset_id

def pre_processing(train_dt, test_dt):
	collection = [
	'country','year','location_type', 'marital_status','uniqueid', 'education_level', 'job_type', 'relationship_with_head']
	non_collection = ['bank_account', 'cellphone_access','gender_of_respondent']
	numerical = ['household_size', 'age_of_respondent']

	[encoding(set(train_dt[cell]), train_dt, test_dt, cell) for cell in collection]
	[encodingOneZero(set(train_dt[cell]), train_dt, test_dt, cell) for cell in non_collection]
	[encodingNumerical(train_dt, test_dt, cell) for cell in numerical]

def encoding(set_, data1, data2, title):
	for ind, i in enumerate(set_):
		data1[title][data1[title] == i] = ind
		data2[title][data2[title] == i] = ind

def encodingOneZero(set_, data1, data2, title):
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

def encodingNumerical(data1, data2, title):
	max_= max(data1[title].max(), data2[title].max())
	data1[title], data2[title] = data1[title]/max_, data2[title]/max_

train_dataset, train_labels, test_dataset, train_dataset_id, test_dataset_id = read_data('data/Train_v2.csv', 'data/Test_v2.csv')

from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from statsmodels.stats.multicomp import pairwise_tukeyhsd 
yes -'cellphone_access - job_type - -country- education_level - household_size'
no - 'gender_of_respondent - '
print(train_dataset.loc[0])
chi2_score, chi_2_p_value = chi2(train_dataset,train_labels)
f_score, f_p_value = f_classif(train_dataset,train_labels)
mut_info_score = mutual_info_classif(train_dataset,train_labels)
print('chi2 score        ', chi2_score)
print('chi2 p-value      ', chi_2_p_value)
print('F - score score   ', f_score)
print('F - score p-value ', f_p_value)
print('mutual info       ', mut_info_score)
print('pairwise_tukeyhsd',pairwise_tukeyhsd)

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


