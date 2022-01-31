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



