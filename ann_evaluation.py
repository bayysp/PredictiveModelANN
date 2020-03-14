import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values

Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#keras.wrappers is use to implement the k-cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
	classifier = Sequential()
	
	classifier.add(Dense(
		units = 6,
		kernel_initializer="uniform",
		activation="relu",
		input_dim = 11
		))
	
	classifier.add(Dense(
		units = 6,
		kernel_initializer="uniform",
		activation="relu"
		))
	
	classifier.add(Dense(
		units = 1,
		kernel_initializer="uniform",
		activation="sigmoid"
		))
	
	classifier.compile(
		optimizer = "adam",
		loss="binary_crossentropy",
		metrics=['accuracy']
		)
	
	return classifier

#this classifier will be use to the 10 different training fold 
#for k-cross validation on 1 test fold
classifier = KerasClassifier(build_fn = build_classifier,
							 batch_size = 10,
							 nb_epoch = 100 )

accuracies = cross_val_score(
		estimator=classifier,
		X = X_train,
		y = Y_train,
		cv=10
		)
#the important variable is cv which mean the number of
#fold in cross validation that we will use

#after we got the accuracies, find the mean
mean = accuracies.mean()
variance = accuracies.std()