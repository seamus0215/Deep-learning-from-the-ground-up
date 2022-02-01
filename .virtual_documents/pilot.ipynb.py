get_ipython().getoutput("pip install tensorflow")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")

import tensorflow as tf


dataset = pd.DataFrame.from_dict({
    'shape': ['round', 'oval'],
    'color': ['yellow', 'green'],
    'produce': ['corn', 'olives']
})

dataset


# Convert parameters to machine learning understandable form.

dataset['c_shape'] = dataset['shape'].apply(lambda x: 1 if x == 'round' else 0)
dataset['c_color'] = dataset['color'].apply(lambda x: 1 if x == 'yellow' else 0)
dataset['c_produce'] = dataset['produce'].apply(lambda x: 1 if x == 'corn' else 0)


dataset


dataset.plot(
    kind='scatter',
    x='c_shape',
    y='c_color',
    c='c_produce',
    colormap='jet'
)


from tensorflow.keras.layers import Dense


# Since we're creating a single neuron layer, the number of units should be 1.

###############################################
# THE NETWORK STRUCTURE
###############################################
single_neuron_layer = Dense(
    units=1,
    input_dim=2,
    activation='sigmoid'
)


###############################################
# THE LOSS FUNCTION
###############################################
loss='binary_crossentropy'


###############################################
# THE OPTIMIZATION ALGORITHM
###############################################
from tensorflow.keras.optimizers import SGD

sgd = SGD()


# Layers in neural nets are connected sequentially

from tensorflow.keras.models import Sequential
single_neuron_model = Sequential()

# Bringing the components into the initialised model above
single_neuron_model.add(single_neuron_layer)
single_neuron_model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
single_neuron_model.summary()


# To train the model with our dataset, we call the fit function on our model

history = single_neuron_model.fit(
    dataset[['c_shape', 'c_color']].values,
    dataset[['c_produce']].values,
    epochs=2500
)


# To make predictions using the model, we call the evaluate function on it

test_loss, test_acc = single_neuron_model.evaluate(
    dataset[['c_shape', 'c_color']],
    dataset[['c_produce']]
)

print(f"Evaluation result on Test Data: Loss = {test_loss}, Accuracy = {test_acc}")



