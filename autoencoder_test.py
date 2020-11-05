#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:22:33 2020

@author: siqisun
"""

#test autodecoder in iris datasets and compare the dimentionality reduction 
#results between pca and autodecoder
#need to tune parameters (e.g. layer #, loss func, optimizer)
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.random import seed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
iris = pd.read_csv("/Users/siqisun/Documents/GitHub/test/Iris.csv")
iris['Species'].replace('Iris-setosa', 0, inplace = True)
iris['Species'].replace('Iris-versicolor', 1, inplace = True)
iris['Species'].replace('Iris-virginica', 2, inplace = True)
#shape (150, 6)


X = iris.iloc[:, 1:-1]
ncol = X.shape[1]
Y = iris.iloc[:, 5]

#plot 3d pca figure
x_reduced = PCA(n_components=3).fit_transform(X)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_reduced[:,0],x_reduced[:,1],x_reduced[:,2],c=Y)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

#plot 2d pca figure
x_reduced = PCA(n_components=2).fit_transform(X)
pca_2d = plt.scatter(x_reduced[:,0],x_reduced[:,1],c=Y)
pca_2d

#set up autoencoder
#split iris data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, random_state = seed(42))
input_dim = Input(shape = (ncol, ))
encoding_dim = 2
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
# DEFINE THE ENCODER LAYER
encoded1 = Dense(3, activation = 'relu')(input_dim)
encoded2 = Dense(encoding_dim, activation = 'relu')(encoded1)
# DEFINE THE DECODER LAYER
decoded1 = Dense(3, activation = 'sigmoid')(encoded2)
decoded2 = Dense(ncol, activation = 'sigmoid')(decoded1)

# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input_dim, decoded2)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
autoencoder.summary()
autoencoder.fit(X_train, X_train, epochs=50, batch_size=100, shuffle=True, validation_data=(X_test, X_test))


# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(input_dim, encoded2)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(X_test)

#2d autoencoder-processed dimentionality reduction figure
ae_2d = plt.scatter(encoded_out[:,0], encoded_out[:,1],c=Y_test)
ae_2d