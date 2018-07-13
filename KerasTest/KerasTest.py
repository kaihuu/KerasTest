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
#from skfeature.function.similarity_based import fisher_score

def oversampling(n, bins, x, y):
    for i in range(n.shape[0]):
        x, y = oversampling_respective(x, y, bins[i], bins[i+1], max(n), n[i])
    return x, y

def oversampling_respective(x, y, startBin, endBin, goalAmount, dataAmount):
    xarray, yarray = getSelectedData(x, y, startBin, endBin)
    #print(xarray)
    if xarray.shape[1] > 0 and yarray.shape[1] > 0:
        for i in range(int(goalAmount) - int(dataAmount)):
        #ランダムにサンプリングしてxに追加
            x, y = randomSampling(x, y, xarray, yarray)
    return x, y

def randomSampling(x, y, xarray, yarray):
    i = np.random.randint(yarray.shape[0])
    #print("yaray: ")
    #print(yarray[i])
    #print()
    #print(yarray.shape)
    x = np.append(x, np.reshape(xarray[i], (1, xarray.shape[1])), axis=0)
    y = np.append(y, np.reshape(yarray[i], (1, yarray.shape[1])), axis=0)
    return x, y

def getSelectedData(x, y, startBin, endBin):
    xarray = np.array([[]])
    yarray = np.array([[]])
    for i, ydata in enumerate(y):
        if ydata >= startBin and ydata < endBin:
            if xarray.shape[1] == 0:
                xi = copy(x[i])
                xarray = np.reshape(xi, (1, xi.shape[0]))
                yi = copy(y[i])
                yarray = np.reshape(yi, (1, 1))
            else:
                #print("ydata")
                #print(ydata.shape)
                #print()
                xarray = np.append(xarray, np.reshape(xi, (1, xi.shape[0])), axis=0)
                yarray = np.append(yarray, np.reshape(ydata, (1, 1)), axis=0)
            
                  
    return xarray, yarray

def superoversampling(x, y):
    binnum = int(math.sqrt(y.shape[0]))
    n, bins, patches = plt.hist(y, bins=binnum) 
    x, y  = oversampling(n, bins, x, y)
    return x, y


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model/model{0:%Y%m%d%H%M%S}'.format(datetime.today()))

if os.path.exists(MODEL_DIR) is False:
	os.mkdir(MODEL_DIR)


rows = dbac.DBAccessor.ExecuteQuery(dbac.DBAccessor.SemanticLinksInputQueryStringV2())

X = np.array(rows)
#print(X)
#print()
#X = np.delete(X, [0, 1], 1)

col = X.shape[1]

mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

for i in range(col):
   X[:,i] = (X[:,i] - mu[i]) / sigma[i]
   #print(sigma[i])


print(X)
print()

rows = dbac.DBAccessor.ExecuteQuery(dbac.DBAccessor.SemanticLinksTrueQueryString())


#node = 10
nodes = np.array([10, 2])
layers = 3
batch_size = 10
log_filepath = './log/'
subdir = datetime.now().strftime("%Y%m%d%H%M%S")

Y = np.array(rows)

#Y = np.delete(Y, [0, 1], 1)

print(Y)
print()





X_train, X_test, Y_train, Y_test =\
    train_test_split(X, Y, train_size=0.8)

X_train, X_validation, Y_train, Y_validation =\
    train_test_split(X_train, Y_train, test_size=0.2)


#print(X_test)


#score = fisher_score.fisher_score(X_train, Y_train.flatten())
#idx = fisher_score.feature_ranking(score)
#num_fea = int(X.shape[1] * 9 / 10)

#selected_features_train = X_train[:, idx[0:num_fea]]
#selected_features_test = X_test[:, idx[0:num_fea]]
#selected_features_validation = X_validation[:, idx[0:num_fea]]

selected_features_train = X_train
selected_features_test = X_test
selected_features_validation = X_validation

input = selected_features_train.shape[1]
node = selected_features_train.shape[1]

selected_features_train, Y_train = superoversampling(selected_features_train, Y_train)
epochs=1000000

kl1 = 0
#5 0.009
kl2 = 1.0 / 16.0
#5 0.05

al1 = 0
al2 = 0

model = Sequential()
model.add(Dense(input_dim=input, units=node, kernel_initializer='he_normal',
                bias_initializer='zeros', kernel_regularizer=regularizers.l1_l2(kl1, kl2)
                , activity_regularizer=regularizers.l1_l2(al1, al2)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))

for i in range(layers):
	model.add(Dense(units=node,kernel_initializer='he_normal',
                bias_initializer='zeros', kernel_regularizer=regularizers.l1_l2(kl1, kl2)
                , activity_regularizer=regularizers.l1_l2(al1, al2)))
	model.add(BatchNormalization())
	model.add(PReLU())
	model.add(Dropout(0.5))


model.add(Dense(units=1, kernel_initializer='he_normal',
                bias_initializer='zeros', kernel_regularizer=regularizers.l1_l2(0, 0)
                , activity_regularizer=regularizers.l1_l2(0, 0)))
#model.add(BatchNormalization())
model.add(Activation('linear'))

model.summary() 
plot_model(model, to_file='model.png', show_shapes=True)
model.compile(loss='mean_squared_error', optimizer=Adam())
#Adam,RMSprop,Adamax,Nadam,Adadelta,Adagrad
#mean_squared_error, 
early_stopping = EarlyStopping(monitor='val_loss',patience=10000, verbose=1, mode='auto')

tensor_board = TensorBoard(log_dir=log_filepath + subdir, histogram_freq=1)

checkpoint = ModelCheckpoint(
	filepath=os.path.join(
		MODEL_DIR,
		'model.hdf5'), monitor='val_loss', verbose=1,
		save_best_only=True, mode='auto')

start = time.time()
hist = model.fit(selected_features_train, Y_train, epochs=epochs, batch_size = batch_size, validation_data=(selected_features_validation, Y_validation)
, callbacks = [checkpoint, tensor_board, early_stopping])


elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

val_loss = hist.history['val_loss']
loss = hist.history['loss']


model = load_model(MODEL_DIR +'/model.hdf5')
eval = model.evaluate(selected_features_test, Y_test)
print()
print(eval)
print()
Ypre = model.predict(selected_features_train, verbose=1)

f.open('url.txt')
url = f.read()
f.close()

requests.post(url, data = json.dumps({
    'text': u'Learning is Finished', # 投稿するテキスト
    'username': u'SHEPHERD', # 投稿のユーザー名
    'icon_emoji': u':finish:', # 投稿のプロフィール画像に入れる絵文字
    'link_names': 1, # メンションを有効にする
}))


print(Ypre)
print()
print(Y_test)
print()
print(Ypre - Y_test)
print()

col = X_test.shape[1]
for i in range(col):
   X_test[:,i] = X_test[:,i] * sigma[i] + mu[i]

data = np.concatenate((X_test, Y_test), axis=1)
data = np.concatenate((data, Ypre), axis=1)
np.savetxt(MODEL_DIR + "/data_{0:%Y%m%d%H%M%S}.csv".format(datetime.today()), data, delimiter=",", fmt='%.2f')

paramdata = np.concatenate((sigma, mu), axis=0)
np.savetxt(MODEL_DIR + "/normalparam_{0:%Y%m%d%H%M%S}.csv".format(datetime.today()), paramdata, delimiter=",", fmt='%.2f')

np.savetxt(MODEL_DIR + "/idx_{0:%Y%m%d%H%M%S}.csv".format(datetime.today()), idx, delimiter=",")
print("param")
print(sigma)
print()
print(mu)
print()
print(paramdata)

sess = tf.Session()
t = sess.run(losses.mean_squared_error(Y_test.T, Ypre.T))
t2 = sess.run(losses.mean_absolute_percentage_error(Y_test.T, Ypre.T))
#print(t[0])
plt.rc('font', family='serif')
fig= plt.figure(1)
plt.plot(val_loss)
plt.plot(loss)
plt.yscale("log")
plt.xlabel('epochs')
plt.ylabel('RMSE[kWh]')
plt.legend(['val_loss', 'loss'], loc='best')
plt.savefig(MODEL_DIR + "/graph2_{0:%Y%m%d%H%M%S}.png".format(datetime.today()))
plt.savefig(MODEL_DIR + "/graph2_{0:%Y%m%d%H%M%S}.eps".format(datetime.today()))
fig, ax = plt.subplots()
ax.scatter(Ypre, Y_test)
lims = [
    #np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
	np.min([0, 0]),
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.title("Test Loss:" + str(format(eval, '.3g')) + "(" + str(format(t2[0], '.3g')) + "%)")
plt.xlabel('predicted')
plt.ylabel('test_data')
#plt.text(float(lims)*0.1,float(lims)*0.1, 'Evaluated Loss:' + str(t) + "%")
plt.savefig(MODEL_DIR + "/graph_{0:%Y%m%d%H%M%S}.png".format(datetime.today()))
plt.savefig(MODEL_DIR + "/graph_{0:%Y%m%d%H%M%S}.eps".format(datetime.today()))

plt.show()

