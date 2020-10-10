#!/usr/bin/env python
# coding: utf-8

# In[43]:


# Подключить пакеты
import numpy as np
import numpy
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema


# In[236]:


#GET DATA

import yfinance as yf

#define the ticker symbol
tickerSymbol = 'AAPL'

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2019-1-25')

#see your data
tickerDf


# In[237]:


plt.plot(tickerDf['Open'])
x = tickerDf['Open']


# In[ ]:





# In[241]:


# Smooth DATA

window_len = 100

s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
w=np.ones(window_len,'d')
y=np.convolve(w/w.sum(),s,mode='valid')
plt.plot(y)


# In[239]:


#Find peaks

from scipy.signal import find_peaks
peaks, _ = find_peaks(y, height=0 , distance=10 ,width = 15 )


# In[240]:


#fig = plt.figure(figsize = (6,8),dpi = 180)
plt.plot(y)
plt.grid()
plt.plot(peaks, y[peaks], "x")
plt.plot(np.zeros_like(y), "--", color="gray")


# In[187]:


#Define lorenzian

def lorentzian( x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)
gam_i = 100
max_x = 100
a = 10
lor = []
for i in range(len(y)):
    lor.append(lorentzian(i,max_x,a,gam_i))
plt.plot(lor)
print(lor[0])


class Lorentzian:
    def __init__(self,array, x0, a, gam):
        self.x0 = x0
        self.a = a
        self.gam = gam
        lor = []
        for i in range(len(array)):
            lor.append(lorentzian(i,x0,a,gam))
        self.lor = lor
           


# In[195]:


#lorentians = [Lorentzian(y,peak,y[peak],5) for peak in peaks]
lorentians = []
for peak in peaks:
    lorentians.append(  Lorentzian(y,peak, y[peak] , 100))


# In[196]:


#PLOT PICTURE
fig = plt.figure(figsize = (6,8),dpi = 180)
plt.plot(y)
plt.grid()
plt.plot(peaks, y[peaks], "x")
plt.plot(np.zeros_like(y), "--", color="gray")
for lorenz in lorentians:
    plt.plot(lorenz.lor)


# In[197]:


fit = [1]*len(y)



for x in range(len(fit)):
    current_lor = [lorenz.lor[x] for lorenz in lorentians ]
    fit[x] = max(current_lor)
    
plt.plot(fit)


# In[198]:


#Get fit for debug
def get_fit(lorentians):
    fit = [1]*len(y)
    for x in range(len(fit)):
        current_lor = [lorenz.lor[x] for lorenz in lorentians ]
        fit[x] = max(current_lor)
    return fit


# In[243]:


fig = plt.figure(figsize = (6,8),dpi = 180)
plt.plot(y)
plt.grid()
plt.plot(peaks, y[peaks], "x")
plt.plot(np.zeros_like(y), "--", color="gray")
plt.plot(fit)


# In[213]:


def get_fit_from_gamma(gamma):
    lorentians = []
    for i in range(len(peaks)):
        lorentians.append(  Lorentzian(y,peaks[i], y[peaks[i]] , gamma[i]))
    fit = get_fit(lorentians) 
    return fit

def plt_diff(fit,y):
    fig = plt.figure(figsize = (6,8),dpi = 180)
    plt.plot(y)
    plt.grid()
    plt.plot(peaks, y[peaks], "x")
    plt.plot(np.zeros_like(y), "--", color="gray")
    plt.plot(fit)


# In[200]:


# ищем ошибку
def mse(A,B,power =2):
    return ((A - B)**power).mean(axis=0)
def sqears(A,B):
    return ((A - B)**2)


# In[275]:


# Ищем ошибку от вариации гаммы

#Array for animation:
list_of_gamma = []


def loss_mse(gamma):
    lorentians = []
    for i in range(len(peaks)):
        lorentians.append(  Lorentzian(y,peaks[i], y[peaks[i]] , gamma[i]))
    fit = get_fit(lorentians) 
    loss =mse(fit,y , power = 4)
    print(f' loss = {loss}')
    print(f'gammas = {gamma}' )
    
    # ВОт это чисто для анимации
    list_of_gamma.append(gamma.copy())
    
    return loss


# In[247]:


# Проверяю почему графики налождились
for g in range(50,100):
    gamma = [g]*9
    lorentians = []
    for i in range(len(peaks)):
        lorentians.append(  Lorentzian(y,peaks[i], y[peaks[i]] , gamma[i]))
    fit = get_fit(lorentians) 
    loss =mse(fit,y , power = 4)
    print(f' loss = {loss}')
    print(f'gammas = {gamma}' )
    plt.plot(fit)
    
plt.show()


# In[276]:


# Launch Optimization

from scipy.optimize import minimize

gamma0 = [10]*4
sol1 = minimize(loss_mse,gamma0, method='COBYLA' , options={'disp': True})


# In[250]:


# Plot result Optimization

plt_diff(get_fit_from_gamma(sol1.x),y)


# # Построим  красивую анимацию:

# In[286]:


from matplotlib.pyplot import savefig
from numpy import logspace

import matplotlib.animation as animation


from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate(i):
    plt.cla()
    plt_diff(get_fit_from_gamma(list_off_gamma[i]),y) 
    

ani = FuncAnimation(plt.gcf(), animate, interval=800 ,frames = 50 )

plt.tight_layout()

HTML(ani.to_jshtml())


# In[282]:


list_of_gamma


# # Чей то код

# In[ ]:





# In[109]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def lorentzian( x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def multi_lorentz( x, params ):
    off = params[0]
    paramsRest = params[1:]
    assert not ( len( paramsRest ) % 3 )
    return off + sum( [ lorentzian( x, *paramsRest[ i : i+3 ] ) for i in range( 0, len( paramsRest ), 3 ) ] )

def res_multi_lorentz( params, xData, yData ):
    diff = [ multi_lorentz( x, params ) - y for x, y in zip( xData, yData ) ]
    return diff

xData, yData = np.loadtxt('HEMAT_1.dat', unpack=True )
yData = yData / max(yData)

generalWidth = 1

yDataLoc = yData
startValues = [ max( yData ) ]
counter = 0

while max( yDataLoc ) - min( yDataLoc ) > .1:
    counter += 1
    if counter > 20: ### max 20 peak...emergency break to avoid infinite loop
        break
    minP = np.argmin( yDataLoc )
    minY = yData[ minP ]
    x0 = xData[ minP ]
    startValues += [ x0, minY - max( yDataLoc ), generalWidth ]
    popt, ier = leastsq( res_multi_lorentz, startValues, args=( xData, yData ) )
    yDataLoc = [ y - multi_lorentz( x, popt ) for x,y in zip( xData, yData ) ]

print (popt)
testData = [ multi_lorentz(x, popt ) for x in xData ]

fig = plt.figure(figsize = (6,8),dpi = 100)
ax = fig.add_subplot( 1, 1, 1 )
ax.plot( xData, yData )
ax.grid()
ax.plot( xData, testData )
plt.show()


# In[111]:


fig = plt.figure(figsize = (6,8),dpi = 100)
ax = fig.add_subplot( 1, 1, 1 )
#ax.plot( xData, yData )
ax.grid()
ax.plot( xData, testData )
plt.show()


# In[115]:


xData = range(len(y))
yData = y / max(y)
yDataLoc = yData
startValues = [ max( yData ) ]

while max( yDataLoc ) - min( yDataLoc ) > .1:
    counter += 1
    if counter > 20: ### max 20 peak...emergency break to avoid infinite loop
        break
    minP = np.argmin( yDataLoc )
    minY = yData[ minP ]
    x0 = xData[ minP ]
    startValues += [ x0, minY - max( yDataLoc ), generalWidth ]
    popt, ier = leastsq( res_multi_lorentz, startValues, args=( xData, yData ) )
    yDataLoc = [ y - multi_lorentz( x, popt ) for x,y in zip( xData, yData ) ]

print (popt)
testData = [ multi_lorentz(x, popt ) for x in range(tickerDf['Open']) ]

fig = plt.figure(figsize = (6,8),dpi = 100)
ax = fig.add_subplot( 1, 1, 1 )
ax.plot( xData, yData )
ax.grid()
ax.plot( xData, testData )
plt.show()


# In[ ]:





# In[ ]:




