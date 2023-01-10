import numpy as np
import pandas as pd
import csv
import jieba #chinese segmentizer
from konlpy.tag import Kkma #korean segmentizer
import random
import pickle

import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
from string import punctuation
from sklearn.naive_bayes import MultinomialNB

punctuation = punctuation + '«»·”…“‘’0123456789。、，—！()▲（）《》②「」）（'

def Segmentizer(x, y):
	"""segmentize chinese sentences to have correct bigrams"""

	segmentized_strings = []

	for i,j in zip(x,y):
		if j == 'chinese':
			segmentized_strings.append(' '.join(jieba.cut(i)))
		else:
			segmentized_strings.append(i)

	return segmentized_strings

def removePunctuation(sentence):
	"""remove punctuation in string so it isn't taken into account in the bigrams"""

	return ''.join([word for word in sentence if word not in punctuation])

def predict(sentence):
	"""removes punctuation in new string and predicts its language"""

	sentence = removePunctuation(sentence)

	x = vectorizer.transform([sentence]).toarray()
	lang = model.predict(x)
	lang = encoder.inverse_transform([round(lang[0])])

	return lang[0]

def cleanArray(x):
	"""removes punctuation from an array"""

	return [removePunctuation(string) for string in x if string not in punctuation]

def stats(y_train,y_test,y_pred):

	"""
	print("Corpus test: ", y_test)
	print("Prédictions : ", [round(x) for x in y_pred])

	print("y axis interception point (b) : ", model.intercept_)
	print("Slope (m) : ", model.coef_)
	"""

	y_true = []
	[y_true.append(y) for x,y in zip(y_test,y_pred) if x == round(y)]
	
	#print("Prédictions correctes: ", [round(x) for x in y_true])

	with open("naive-bayes-model-stats.txt", "w") as file:
		file.write(f'Nombre de prédictions correctes: {len(y_true)}\nPourcentage de prédictions correctes : {round((len(y_true)/len(y_pred))*100)}%\nAccuracy score : {model.score(x_test,y_test)}')

if __name__ == '__main__':

	data = pd.read_csv('sentences_CH_KR.tsv', sep='\t')

	print(data["language"].value_counts())

	x,y = np.array([string.lower() if isinstance(string, str) else string for string in data['sentence']]),np.array(data['language'])
	
	x = cleanArray(x)
	x = Segmentizer(x,y)

	vectorizer = CountVectorizer(ngram_range=(2,2), max_features=10000)

	# 'fr' = 1, 'en'= 0
	encoder = LabelEncoder()
	y = encoder.fit_transform(y)

	X = vectorizer.fit_transform(x)
	x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

	"""attempt at saving the best model possible"""
	best_accuracy = 0

	for __ in range(100):

		x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
		model = MultinomialNB().fit(x_train,y_train)
		accuracy = model.score(x_test,y_test)
		print("Accuracy : ", accuracy )

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			#saving model
			with open("naive-bayes-model.pickle", "wb") as file:
				pickle.dump(model, file)

	pickle_input = open("naive-bayes-model.pickle", "rb")
	model = pickle.load(pickle_input)

	y_pred = model.predict(x_test)

	"""write stats into linear-regression-model.txt"""
	stats(y_train,y_test,y_pred)

	sentence = input("Type a sentence : ")

	print("The sentence is in :", predict(sentence))