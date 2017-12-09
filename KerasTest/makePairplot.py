import DBAccessor as dbac
import csv
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation
from keras.optimizers import SGD,Adam,RMSprop,Adamax,Nadam,Adadelta,Adagrad
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint
from keras.layers.core import Dropout
from keras.initializers import TruncatedNormal
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras import regularizers
import matplotlib.pyplot as plt
import sys
from keras.utils import plot_model
from keras import losses
import time
import os
import requests
import json
import math
from copy import copy
import plotly.plotly as py
import plotly.figure_factory as ff
import pandas as pd
import plotly


plotly.tools.set_credentials_file(username='kaihuu', api_key='BhY6sn42ToQvN62HR1lL')
rows = dbac.DBAccessor.ExecuteQuery(dbac.DBAccessor.SemanticLinksInputQueryStringTest())

X = np.array(rows)
#print(X)
#print()
#X = np.delete(X, [0, 1], 1)


col = X.shape[1]

mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

#for i in range(col):
   #X[:,i] = (X[:,i] - mu[i]) / sigma[i]
   #print(sigma[i])


rows = dbac.DBAccessor.ExecuteQuery(dbac.DBAccessor.SemanticLinksTrueQueryString())

input = X.shape[1]
node = X.shape[1]
#node = 10
nodes = np.array([10, 2])
layers = 5
batch_size = 100
log_filepath = './log/'
subdir = datetime.now().strftime("%Y%m%d%H%M%S")

Y = np.array(rows)

#dataframe = pd.DataFrame(np.concatenate((X, Y), axis=1), columns=['time', 'avgspeed', 'stdspeed', 'maxspeed', 'energy'])
dataframe = pd.DataFrame(np.concatenate((X, Y), axis=1), columns=['time', 'avgspeed', 'avgspeed2', 'avgaccuracy','avgtheta', 'TEMPERATURE', 'FIRSTGIDS', 'energy'])

fig = ff.create_scatterplotmatrix(dataframe, diag='histogram', width=1920, height=1920, title='InputData')

py.plot(fig, filename='test{0:%Y%m%d%H%M%S}'.format(datetime.today()))