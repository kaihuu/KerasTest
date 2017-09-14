import DBAccessor as dbac
import csv
import datetime
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD,Adam
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout
from keras.initializers import TruncatedNormal
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

rows = dbac.DBAccessor.ExecuteQuery("""SELECT groupa.avgspeed, groupa.stdspeed, groupa.maxspeed, groupa.minspeed, groupa.avgac, groupa.stdac
FROM(
SELECT TRIP_ID, AVG(SPEED) as avgspeed, STDEV(SPEED) as stdspeed, MAX(SPEED) as maxspeed, MIN(SPEED) as minspeed,
AVG(ACCURACY) as avgac, STDEV(ACCURACY) as stdac
FROM CORRECTED_GPS_modified, SEMANTIC_LINKS, TRIPS_modified
WHERE SEMANTIC_LINKS.SEMANTIC_LINK_ID = 2 AND SEMANTIC_LINKS.LINK_ID = CORRECTED_GPS_modified.LINK_ID AND TRIPS_modified.SENSOR_ID = CORRECTED_GPS_modified.SENSOR_ID
AND CORRECTED_GPS_modified.JST >= TRIPS_modified.START_TIME AND CORRECTED_GPS_modified.JST <= TRIPS_modified.END_TIME AND SPEED IS NOT NULL
GROUP BY TRIP_ID) AS groupa, 
(SELECT gps.TRIP_ID
FROM
(
SELECT LEAFSPY_TIME_INTERVAL_View.*, LEAFSPY_RAW2.GIDS
FROM LEAFSPY_TIME_INTERVAL_View,LEAFSPY_RAW2
WHERE LEAFSPY_TIME_INTERVAL_View.TRIP_ID = LEAFSPY_RAW2.TRIP_ID AND LEAFSPY_TIME_INTERVAL_View.START_TIME = LEAFSPY_RAW2.DATETIME
) AS LEAFSPY,
(
SELECT TRIP_ID, MIN(JST) as minjst, MAX(JST) as maxjst
FROM CORRECTED_GPS_modified, SEMANTIC_LINKS, TRIPS_modified
WHERE CORRECTED_GPS_modified.LINK_ID = SEMANTIC_LINKS.LINK_ID AND SEMANTIC_LINKS.SEMANTIC_LINK_ID = 2
AND CORRECTED_GPS_modified.SENSOR_ID = TRIPS_modified.SENSOR_ID  AND CORRECTED_GPS_modified.JST >= TRIPS_modified.START_TIME
AND CORRECTED_GPS_modified.JST <= TRIPS_modified.END_TIME AND SPEED IS NOT NULL
GROUP BY TRIP_ID
) AS gps
WHERE LEAFSPY.START_TIME >= gps.minjst AND LEAFSPY.START_TIME <= gps.maxjst
GROUP BY gps.TRIP_ID) as groupb
WHERE groupa.TRIP_ID = groupb.TRIP_ID
ORDER BY groupa.TRIP_ID

""")

X = np.array(rows)

rows = dbac.DBAccessor.ExecuteQuery("""SELECT SUM(GIDS_DIFFERENCE)
FROM
(
SELECT LEAFSPY_TIME_INTERVAL_View.*, LEAFSPY_RAW2.GIDS
FROM LEAFSPY_TIME_INTERVAL_View,LEAFSPY_RAW2
WHERE LEAFSPY_TIME_INTERVAL_View.TRIP_ID = LEAFSPY_RAW2.TRIP_ID AND LEAFSPY_TIME_INTERVAL_View.START_TIME = LEAFSPY_RAW2.DATETIME
) AS LEAFSPY,
(
SELECT TRIP_ID, MIN(JST) as minjst, MAX(JST) as maxjst
FROM CORRECTED_GPS_modified, SEMANTIC_LINKS, TRIPS_modified
WHERE CORRECTED_GPS_modified.LINK_ID = SEMANTIC_LINKS.LINK_ID AND SEMANTIC_LINKS.SEMANTIC_LINK_ID = 2
AND CORRECTED_GPS_modified.SENSOR_ID = TRIPS_modified.SENSOR_ID  AND CORRECTED_GPS_modified.JST >= TRIPS_modified.START_TIME
AND CORRECTED_GPS_modified.JST <= TRIPS_modified.END_TIME AND SPEED IS NOT NULL
GROUP BY TRIP_ID
) AS gps
WHERE LEAFSPY.START_TIME >= gps.minjst AND LEAFSPY.START_TIME <= gps.maxjst
GROUP BY gps.TRIP_ID
""")


Y = np.array(rows)

X_train, X_test, Y_train, Y_test =\
    train_test_split(X, Y, train_size=0.8)

X_train, X_validation, Y_train, Y_validation =\
    train_test_split(X_train, Y_train, test_size=0.2)

#print(X_test)

epochs=4000

model = Sequential()
model.add(Dense(input_dim=6, units=5, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(units=4, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(units=3, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(units=2, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(units=1, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(Activation('linear'))

model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

hist = model.fit(X_train, Y_train, epochs=epochs, batch_size = 1, validation_data=(X_validation, Y_validation)
, callbacks = [early_stopping])

val_loss = hist.history['val_loss']
loss = hist.history['loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(val_loss)
plt.plot(loss)
plt.xlabel('epochs')
plt.legend(['val_loss', 'loss'], loc='best')
plt.show()

