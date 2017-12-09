import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy

def oversampling(n, bins, x, y):
    for i in range(n.shape[0]):
        x, y = oversampling_respective(x, y, bins[i], bins[i+1], max(n), n[i])
    return x, y

def oversampling_respective(x, y, startBin, endBin, goalAmount, dataAmount):
    xarray, yarray = getSelectedData(x, y, startBin, endBin)
    if xarray.shape[0] > 0 and yarray.shape[0] > 0:
        for i in range(int(goalAmount) - int(dataAmount)):
        #ランダムにサンプリングしてxに追加
            x, y = randomSampling(x, y, xarray, yarray)
    return x, y

def randomSampling(x, y, xarray, yarray):
    i = np.random.randint(xarray.shape[0])
    x = np.append(x, xarray[i])
    y = np.append(y, yarray[i])
    return x, y

def getSelectedData(x, y, startBin, endBin):
    xarray = np.array([])
    yarray = np.array([])
    
    for (xdata, ydata) in zip(x, y):
        if xdata >= startBin and xdata < endBin:
            xarray = np.append(xarray, xdata)
            yarray = np.append(yarray, ydata)
            #print(ydata)        
    return xarray, yarray

def superoversampling(x, y):
    binnum = int(math.sqrt(x.shape[0]))
    n, bins, patches = plt.hist(x, bins=binnum) 
    x, y  = oversampling(n, bins, x, y)
    return x, y

# 平均 50, 標準偏差 10 の正規乱数を1,000件生成
x = np.random.normal(50, 10, 10000)
y = copy(x)
xarray = np.array([[]])
print(xarray.shape[1])
print(y)
print(y.shape)
y = np.reshape(y, (1, y.shape[0]))
print(y)
print(y.shape)
xdata, ydata = superoversampling(x, y)
plt.hist(xdata , bins=int(math.sqrt(xdata.shape[0]))) 
plt.show()

