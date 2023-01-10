import numpy as np
import pandas as pd
import csv
import pickle
import re

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics

def predict(sentence):
	"""predict language of new input"""

	x = vectorizer.transform([sentence]).toarray() # sentence to vector
	lang = model.predict(x) # predict y

	return lang[0].title() # return y

if __name__ == '__main__':

	data = pd.read_csv('sentences.tsv', sep='\t') # load dataset

	print(data["language"].value_counts()) # print number of sentence per language

	x,y = np.array([string.lower() if isinstance(string, str) else string for string in data['sentence']]),np.array(data['language']) # put all text in lowercase

	vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1), max_features=10000) # vectorizer

	X = vectorizer.fit_transform(x)

	x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1) # split corpus into test and train with test equals to 10% of corpus

	'''attempt at saving the best model possible'''
	best_accuracy = 0

	for __ in range(5): # we'll train the model 5 times

		x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
		model = KNeighborsClassifier(n_neighbors=2).fit(x_train,y_train) # training
		accuracy = model.score(x_test,y_test) 
		print("Accuracy :", accuracy) # get accuracy score

		if accuracy > best_accuracy: # if our new accuracy is greater than our saved best accuracy
			best_accuracy = accuracy # we make this new accuracy our best accuracy
			#saving model
			with open("knn-unigram-model.pickle", "wb") as file:
				pickle.dump(model, file) # and save it to a file in binary format

	pickle_input = open("knn-unigram-model.pickle", "rb")
	model = pickle.load(pickle_input) # we then read and load this file

	y_pred = model.predict(x_test) # make a prediction

	'''write stats into linear-regression-model.txt'''
	with open("knn-unigram-model-stats.txt", "w") as file:
		file.write(f'Scores:\n{metrics.classification_report(y_test,y_pred)}')

	sentence = input("Type a sentence : ")

	print(f"The sentence is in {predict(sentence)}.")