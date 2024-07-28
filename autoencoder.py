

from math import sqrt
from numpy import concatenate
import tensorflow as tf
import math
import os
                                                              


import numpy as np

from tensorflow import keras
#from tensorflow.keras import backend as K
from keras import backend as K

import random
#from keras.utils import np_utils

from sklearn.utils import shuffle

X = np.array([[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0]])
y = np.array([[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0]])

COST_FUNCTION = "mean_squared_error"
    #design Network                                                                                            
model = keras.Sequential()
model.add(keras.layers.Dense(3, input_dim=8, activation='tanh'))
# Output layer                                                                                  
model.add(keras.layers.Dense(8, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.08)
model.compile(loss=COST_FUNCTION, optimizer = opt) # ,metrics=['mae'])
print(model.summary())

#print(X.shape)
#print(y.shape)

history = model.fit(X, y, validation_data=(X,y),epochs=800, batch_size=1,
                    verbose=2, shuffle=False)
model.save('autoenc'+'.h5')
predictions = model.predict(X)
predictions=np.round(predictions)
print("Predictions are\n",predictions)
print("Model Weights")
print (model.get_weights())

# This works in jupyter notebook only
get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])

layer_output = get_1st_layer_output(X)[0]
print("Hidden output: ", layer_output)
new_input=layer_output

# Build model to predict from this compressed input (of 3 values) with a single or more layers
# neural network

# define the new model
new_model = keras.Sequential([keras.layers.Dense(3, input_dim=3, activation='tanh'),
                              keras.layers.Dense(8, activation='sigmoid')])

# compile the model 
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
history = new_model.fit(new_input, y, epochs=800, batch_size=1,
                        verbose=2, shuffle=False)

# make predictions on the original input data
new_predictions = new_model.predict(layer_output)
print('Predictions:\n', new_predictions)

# round to the nearest integer
print("Rounded predictions:\n", np.round(new_predictions))