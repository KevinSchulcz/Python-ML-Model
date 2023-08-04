# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:35:11 2022

@author: kevin
"""

#%% Lab 8: Number Identification 

#%% Imports

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout #
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.optimizers import SGD
import tensorflow.keras as keras

#%% Load Data
(trainX, trainY), (testX, testY) = mnist.load_data()
print(trainX.shape)
print(trainY.shape)

for i in range(10):
    plt.figure()
    plt.imshow(trainX[i]) #takes a 2D array and shows it as an image
    plt.title(f"{trainY[i]}")
    plt.show
    
#Capping training size
train_size = 20000
trainX = trainX[:train_size]
trainY = trainY[:train_size]

#%% Pre-Processing to make it easier to process

#reshape matrices so that it will work with Conv2D function (just made to recognize matrices like this)
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)) 
testX = testX.reshape((testX.shape[0], 28, 28, 1)) 

# one hot encoding
print("before encoding", trainY[0])
trainY = keras.utils.to_categorical(trainY)
testY = keras.utils.to_categorical(testY)
print("after encoding", trainY[0])

#%% Final Pre-processing step

#convert to float
trainX = trainX.astype('float32')
testX = testX.astype('float32')

#Normalize
trainX_norm = trainX/255.0
testX_norm = testX/255.0

#%% Create Model

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.4)) # adds dropout layer to prevent memorization (by cutting random connections each run 0.4 = 40% of connections)
model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

#%% Compile Model

opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#%% Train Model

history = model.fit(trainX_norm, trainY, epochs=10, batch_size=32, validation_data=(testX_norm, testY))

#%% Show Accuracy

_, acc = model.evaluate(testX_norm, testY)
print('> %.3f' % (acc * 100.00))

#%% Plot Loss Progress of Both Train and Test

plt.figure()
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')
plt.legend()
plt.show()

#%% Plot Accuracy Progress of Both Train and Test 

plt.figure()
plt.title('Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')
plt.legend()
plt.show()

#%% Using Learning to Predict 

predictionY = model.predict(testX_norm)
print(predictionY.shape)
print(np.argmax(predictionY[0]))

#%% Show Test Numbers (from 0-10) With Answer and Prediction

for i in range(0,10):
    plt.figure()
    plt.imshow(testX_norm[i])
    plt.title(f"Answer: {np.argmax(testY[i])} | Prediction: {np.argmax(predictionY[i])}") # f"" format string type, {} in a print means to insert code
    plt.show()
    
#%% To K-fold 

from sklearn.model_selection import KFold

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dropout(0.4)) # adds dropout layer to prevent memorization (by cutting random connections each run 0.4 = 40% of connections)
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    
    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        
        trainX = dataX[train_ix]
        trainY = dataY[train_ix]
        testX = dataX[test_ix]
        testY = dataY[test_ix]
        
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))
        
        _, acc = model.evaluate(testX, testY)
        print('> %.3f' % (acc * 100.00))
        
        #Store scores
        scores.append(history)
        histories.append(history)
    return scores, histories

scores, histories = evaluate_model(trainX_norm, trainY)

#%% Plot Loss and Accuracy of All K-Folds

for i in range(len(histories)):
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(histories[i].history['loss'], color='blue', label='train')
    plt.plot(histories[i].history['val_loss'], color='orange', label='test')
    
    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(histories[i].history['accuracy'], color='blue', label='train')
    plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.legend()

plt.show()

  