import numpy as np
import pandas as pd
import csv
import random
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import sklearn.metrics as metrics
from string import punctuation

def predict(sentence):
	"""removes punctuation in new string and predicts its language"""

	x = vectorizer.transform([sentence]).toarray() # vectorize sentence
	lang = model.predict(x) # predict y from x

	return lang[0] # return y

def clean_data(x):
	'''function to clean dataset'''
	data = []
	for line in x:
		line = re.sub(r'[!@#$(),"%^*?:;~`0-9]', ' ', line)
		line = re.sub(r'[\[\]]', ' ', line)
		line.lower()
		data.append(line)

	return data

if __name__ == '__main__':

	data = pd.read_csv('17langues.csv', sep=',')

	print(data["Language"].value_counts())

	x,y = np.array(data['Text']),np.array(data['Language'])

	vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1), max_features=10000)

	x = clean_data(x)
	X = vectorizer.fit_transform(x)

	x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

	"""attempt at saving the best model possible"""
	best_accuracy = 0

	for __ in range(100):

		x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
		model = MultinomialNB().fit(x_train,y_train)
		accuracy = model.score(x_test,y_test)
		print("Accuracy :", accuracy)

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			#saving model
			with open("multinomial-naive-bayes-model.pickle", "wb") as file:
				pickle.dump(model, file)

	pickle_input = open("multinomial-naive-bayes-model.pickle", "rb")
	model = pickle.load(pickle_input)

	y_pred = model.predict(x_test)

	"""write stats into linear-regression-model.txt"""
	with open("multinomial-naive-bayes-model-stats.txt", "w") as file:
		file.write(f'Scores:\n {metrics.classification_report(y_test,y_pred)}')


	cm = metrics.confusion_matrix(y_test, y_pred)

	plt.figure(figsize=(15,10))
	sns.heatmap(cm, annot = True)
	plt.savefig('confusion_matrix.png')
	plt.show()

	sentence = input("Type a sentence : ")

	print(f'The sentence is in {predict(sentence)}.')