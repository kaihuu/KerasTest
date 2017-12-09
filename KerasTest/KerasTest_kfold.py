import DBAccessor as dbac
import csv
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split,StratifiedKFold
from keras.models import Sequential
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



MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

if os.path.exists(MODEL_DIR) is False:
	os.mkdir(MODEL_DIR)
    
fold_num = 5

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)



rows = dbac.DBAccessor.ExecuteQuery(dbac.DBAccessor.SemanticLinksInputQueryString())

X = np.array(rows)
print(X)
print()
X = np.delete(X, [0, 1], 1)


col = X.shape[1]

mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

for i in range(col):
   X[:,i] = (X[:,i] - mu[i]) / sigma[i]
   print(sigma[i])


print(X)


rows = dbac.DBAccessor.ExecuteQuery(dbac.DBAccessor.SemanticLinksTrueQueryString())

input = X.shape[1]
node = X.shape[1]
nodes = np.array([10, 2])
layers = 3
batch_size = 5
log_filepath = './log/'
subdir = datetime.now().strftime("%Y%m%d%H%M%S")

Y = np.array(rows)

Y = np.delete(Y, [0, 1], 1)

print(Y)
# X[train], X_test, Y[train], Y_test =\
#     train_test_split(X, Y, train_size=0.8)

# X[train], X_validation, Y[train], Y_validation =\
#     train_test_split(X[train], Y[train], test_size=0.2)

#print(X_test)


kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
cvscores = []


epochs=50000

kl1 = 0
#5 0.009
kl2 = 0
#5 0.05

al1 = 0
al2 = 0
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(input_dim=input, units=node, kernel_initializer='he_normal',
                    bias_initializer='zeros', kernel_regularizer=regularizers.l1_l2(kl1, kl2)
                    , activity_regularizer=regularizers.l1_l2(al1, al2)))
    model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.2))

    for i in range(layers):
        model.add(Dense(units=node,kernel_initializer='he_normal',
                    bias_initializer='zeros', kernel_regularizer=regularizers.l1_l2(kl1, kl2)
                    , activity_regularizer=regularizers.l1_l2(al1, al2)))
        model.add(BatchNormalization())
        model.add(PReLU())
        #model.add(Dropout(0.5))


    model.add(Dense(units=1, kernel_initializer='he_normal',
                    bias_initializer='zeros', kernel_regularizer=regularizers.l1_l2(kl1, kl2)
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
            'model_{epoch:02d}_vloss{val_loss:.2f}.hdf5'),
            save_best_only=True)

    start = time.time()
    X[train], X_validation, Y[train], Y_validation =\
        train_test_split(X[train], Y[train], test_size=0.2)

    hist = model.fit(X[train], Y[train], epochs=epochs, batch_size = batch_size, validation_data=(X_validation, Y_validation)
    , callbacks = [checkpoint, tensor_board])


    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    val_loss = hist.history['val_loss']
    loss = hist.history['loss']

    eval = model.evaluate(X_test, Y_test)
    print()
    print(eval)
    print()
    Ypre = model.predict(X_test, verbose=1)


    print(Ypre)
    print()
    print(Y_test)
    print()
    #print(Ypre - Y_test)
    #print()
    sess = tf.Session()
    t = sess.run(losses.mean_absolute_percentage_error(Y_test.T, Ypre.T))
    print(t[0])
    plt.rc('font', family='serif')
    fig= plt.figure(1)
    plt.plot(val_loss)
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('RMSE[kWh]')
    plt.legend(['val_loss', 'loss'], loc='best')

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
    plt.xlabel('predicted')
    plt.ylabel('test_data')
    plt.text(0,0, 'Evaluated Loss:' + str(t) + "%")

    plt.show()

