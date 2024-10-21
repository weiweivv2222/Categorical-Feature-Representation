import pandas as pd
import os
import numpy as np
from numpy import interp
import sys
import random
import tensorflow as tf
seed=42
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

import category_encoders as ce
from keras.layers import Embedding
from keras.models import Sequential

from keras import backend 
from keras import models
from keras import layers
from keras.layers import Input

#%% create a NN model
# Create function returning a compiled network
def create_network(number_of_features):
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    #network.add(layers.Dense(units=32, activation='relu', input_shape=(number_of_features,)))
    # Add input layer (using Input to handle the input shape)
    network.add(Input(shape=(number_of_features,)))

    # Add fully connected layers with ReLU activation functions
    network.add(layers.Dense(units=32, activation='relu'))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=32, activation='relu'))
    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu'))
    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile neural network
    network.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer='rmsprop', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return network


