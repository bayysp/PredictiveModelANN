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










 