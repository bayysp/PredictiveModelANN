# -*- coding: utf-8 -*-

#part 1 - data preprocessing

#step 1 - import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#step 2 - import dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#check 1 by 1 in csv and find the independent variable
#after that put in into new variable

X = dataset.iloc[:, 3:13].values

#put dependent variable to new variable
Y = dataset.iloc[:, 13].values

#splitting the dataset into train and test set
#before splitting data, dont forget to encode the categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#there is 2 variable should encoding (country as X_1 and gender as X_2)
labelencoder_X_1 = LabelEncoder()
#because country are on index 1 so write in in the code
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
#after execute, value X index 1 will change into 0(france),1(germany) and 2(spain)

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#after execute , gender value will change into 0(female) and 1 (male)

#because in country variable is no ordinal value (spain is not high than france, etc)
#change the value using dummy variable with onehotencoder
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#the value of X will be change to float64
#the column of country will be deleted and there is a new 3 dummy variables(index 0,1,2) corresponding to the country

X = X[:, 1:] #remove the first column ()
# first ':' means is all the lines, 1 means second index in column , second ':' until the last column
# we remove 1 dummy var because the formula is : sum_of_dummy_var = sum_of_category_in_var - 1

#do data splitting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #test_size 0.2 means 20% of data be test dataset

#do feature scalling
#feature scalling means make a scala on the variables be same
#if the scala each variable is not same, it will make problem to ML model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#------------------- PART 1 clear--------------------

#part 2 build ANN

#import keras
import keras
#import 2 modules
#sequential module is to initialize neural network
#dense module to build layers neural network
from keras.models import Sequential
from keras.layers import Dense

#initialize NN using sequential, defining sequence of layer
classifier = Sequential() #we use classification because were going to predict tje tested result
#using classifier

#add first layer(input layer) and first hiddent layer
#because we have 11 independent var, then we will make 11 input layer

#build a first hidden layer with the activation function using rectifier function
#sigmoid function is good for output layer

classifier.add(Dense(
		units = 6,
		kernel_initializer="uniform",
		activation="relu",
		input_dim = 11
		))
#first parameter is number of nodes what we want to add
#second parameter is init to randomly small number close to 0
#third parameter is activation-function (relu is rectifier function)
#fourth parameter is sum of nodes input layer

#execute line 80, at the same time, input and first hidden layer were added


#based on Ku rill's tutorial, we make a 6 nodes in hidden layer 
#(11(as a independent var) - 1(output var)) = 6

#next step is add second hidden layer
classifier.add(Dense(
		units = 6,
		kernel_initializer="uniform",
		activation="relu"
		))

#adding output layer (final layer), sum of units is 1 because we just have 1 output
classifier.add(Dense(
		units = 1,
		kernel_initializer="uniform",
		activation="sigmoid"
		))
#activation we use sigmoid because we want to make a probabilistic output

#NB we can use softmax as activation, softmax is sigmoid but it applied if we have more than 2 categories

# ---------- PART 2 build ANN architecture finish -----------

#part 3 is compile ANN, basically applying stochastic gradient descent

#compiling ANN
classifier.compile(
		optimizer = "adam",
		loss="binary_crossentropy",
		metrics=['accuracy']
		)

#first parameter (optimizer = adam )is type of stocastic gradient descent which 
#very efficient to find optimum weight
#second parameter is loss function, loss func was the sum of squared errors
#if we have binary outcome, logaritmic loss function called binary_crossentropy, 
#if more than 2 , called it by categorical_crossentropy
#third parameter is metric (the criterion that we choose to evalute the model) 
#to improve the model performance
#we use accuracy criterion

#we complete build ann but not making connection to training set
	
#----------- PART 3 is finished-----------------

#part 4 is we gonna see algorithm in action

#choose the number of epochs (the number of times we are training our a and n on the whole training set)
#going to see how stochastic gradient descent in action
#gonna see how our a and n model is trained
#and how to improving accuracy at each round that is at epoch

classifier.fit(
		X_train,
		Y_train,
		batch_size=10,
		epochs=100)
#first parameter is the dataset that we want to train
#second parameter is output of dependent variable that we train

#----------- PART 4 is finished -----------------

#part 5 is we gonna make an prediction with classifier

Y_predict = classifier.predict(X_test)

#after we get the predict value, convert it into boolean with threshold
Y_predict = (Y_predict > 0.5)

#after we make an prediction, we need make a confusion matrix to find the accuracy of testing data

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
#to find the accuracy write this in the console : (TP+FN)/sum_of_test_data "(1547+142)/2000"

#------------------ COMPLETE -------------------

#example, there is a new data;
"""
	'CreditScore' : 600,
	'Geography' : 'France',
	'Gender' : 'Male',
	'Age' : 40,
	'Tenure' : 3,
	'Balance' : 60000,
	'NumOfProducts' : 2,
	'HasCrCard' : 1,
	'IsActiveMember' : 1,
	'EstimatedSalary' : 50000
"""

#if you want to add new test data, there is some step that should you do

#because in testing we have 2 dummy var, we should compare it based on X variable
#look into X var, if the country is france, then the dummy var is 0 and 0
#and for the gender compare it too , if female the value is 0 and the male is 1
#dont forget do standarization using StandarScaller

#this is the code :
""" new_prediction = classifier.predict(
		sc.transform(
				np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
				)
		)
new_prediction = (new_prediction > 0.5)
"""

# PART 5 make a evaluating of ANN
#we need to make a evaluating because if we trained the dataset for several times,
#the accuray will decreased (it make a some variance of accuracy, that's not good for our model)

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

#make a classifier
classifier = KerasClassifier(
		build_fn=build_classifier,
		batch_size = 10,
		nb_epoch = 100)

accuracies = cross_val_score(
		estimator=classifier,
		X=X_train,
		y=Y_train,
		cv=10,
		n_jobs=1)

mean = accuracies.mean()
variance= accuracies.std()
#estimator is the object implementing fit (will be trained)
#cv is the number of folds that we will use
#n_jobs is the number of cpu will used, -1 use all cpu,
#if you want to use -1, you need to install tensorflow-gpu