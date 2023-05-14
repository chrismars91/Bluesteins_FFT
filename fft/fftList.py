import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt 
import time
import numpy as np
# from cmake_example import Bluestein as blue_cpp 
# library I made to test my cpp. Used pybind
import math
from numba import jit
pi = np.pi

#%%

def safe_zero(m,n):
    a_main = []
    for i in range(m):
        a_inner = []
        for k in range(n):
            a_inner.append(0)
        a_main.append(a_inner) 
    return a_main


def ifft_list(x):
    N = len(x)
    M = [[complex(1,0),complex(1,0)],
          [complex(1,0),complex(-1,0)]]    
    M = np.array(M)
    ROWS = 2
    COLS = N//2
    COLS_O2 = COLS//2 
    Ax = safe_zero(ROWS,COLS)
    for j in range(ROWS):
        for k in range(COLS):       
            for i in range(ROWS):            
                Ax[j][k] +=  M[j][i] * x[i*COLS+k]    
    for i in range(int(np.log2(N)-1)):
        J3 = safe_zero(2*ROWS,COLS_O2)
        for i in range(ROWS):
            t = np.exp(-i*1j*pi/ROWS)   
            for k in range(COLS_O2):
                e = Ax[i][k]
                o = Ax[i][k+COLS_O2]
                J3[i][k] = e+t*o
                J3[i+ROWS][k] = e-t*o              
        Ax = J3
        ROWS = ROWS*2
        COLS = COLS//2  
        COLS_O2 = COLS//2
    return [Ax[i][0]/N for i in range(N)] 
   

def fft_list_blue(x,y):
    N = len(x)
    M = [[complex(1,0),complex(1,0)],
          [complex(1,0),complex(-1,0)]]    
    M = np.array(M)
    ROWS = 2
    COLS = N//2
    COLS_O2 = COLS//2 
    Ax = safe_zero(ROWS,COLS)
    Ay = safe_zero(ROWS,COLS)
    for j in range(ROWS):
        for k in range(COLS):       
            for i in range(ROWS):            
                Ax[j][k] +=  M[j][i] * x[i*COLS+k]   
                Ay[j][k] +=  M[j][i] * y[i*COLS+k] 
    for i in range(int(np.log2(N)-1)):
        Jx = safe_zero(2*ROWS,COLS_O2)
        Jy = safe_zero(2*ROWS,COLS_O2)
        for i in range(ROWS):
            t = np.exp(i*1j*pi/ROWS)   
            for k in range(COLS_O2):
                e = Ax[i][k]
                o = Ax[i][k+COLS_O2]
                Jx[i][k] = e+t*o
                Jx[i+ROWS][k] = e-t*o  
                e = Ay[i][k]
                o = Ay[i][k+COLS_O2]
                Jy[i][k] = e+t*o
                Jy[i+ROWS][k] = e-t*o                  
        Ax = Jx
        Ay = Jy
        ROWS = ROWS*2
        COLS = COLS//2  
        COLS_O2 = COLS//2
    return np.array([Ax[i][0]*Ay[i][0] for i in range(N)])   
  

def bluestein_list(signal):
    n = len(signal)
    l = int(2 ** np.ceil(np.log2(2 * n + 1)))
    U_l,V_l,V_star = [0]*l,[0]*(l+1),[0]*n
    for i in range(n):
        comp = np.exp( (1j*pi*i**2)/n )
        V_star[i] = 1/comp
        U_l[i] = signal[i]/comp
        V_l[i] = comp
        V_l[l-i] = comp 
    V_l.pop()
    uv_l = fft_list_blue(U_l,V_l)
    ift = ifft_list(uv_l) 
    return [V_star[i]*ift[i] for i in range(len(V_star))]

@jit(nopython=True)
def bluestein_np(signal):
    n = signal.shape[0]
    l = int(2 ** np.ceil(np.log2(2 * n + 1)))
    xi = np.exp(1j*pi*(np.arange(n)**2)/n).astype(np.cdouble)
    V_star = 1/xi
    U_l, V_l = np.zeros(l, dtype=np.cdouble), np.zeros(l, dtype=np.cdouble)
    U_l[:n] = signal/xi
    V_l[0:n] = xi
    V_l[l-n+1:] = xi[1:][::-1]
    uv_l = fft_np_blue(U_l, V_l)
    ift = ifft_np(uv_l)
    return V_star*ift[0:n]

@jit(nopython=True)
def fft_np_blue(x,y):
    N = x.shape[0]
    M = np.array([[complex(1, 0), complex(1, 0)],
                  [complex(1, 0), complex(-1, 0)]], dtype=np.cdouble)
    Ax = np.dot(M, x.reshape((2, N//2)))
    Ay = np.dot(M, y.reshape(2, N//2))
    for _ in range(int(np.log2(N)-1)):
        ROWS,COLS_O2 = Ax.shape[0],Ax.shape[1]//2      
        terms = np.exp(1j*pi*np.arange(ROWS)/ROWS).reshape(ROWS,1)
        X_even = Ax[:,:COLS_O2]
        X_odd = Ax[:,COLS_O2:]
        Ax = np.zeros(shape=(ROWS*2,COLS_O2), dtype=np.cdouble)    
        Ax[0:ROWS] = X_even + terms * X_odd      
        Ax[ROWS:] = X_even - terms * X_odd        
        X_even = Ay[:,:COLS_O2]
        X_odd = Ay[:,COLS_O2:]
        Ay = np.zeros(shape=(ROWS*2,COLS_O2), dtype=np.cdouble)    
        Ay[0:ROWS] = X_even + terms * X_odd      
        Ay[ROWS:] = X_even - terms * X_odd   
    return Ax.ravel() * Ay.ravel()

@jit(nopython=True)
def ifft_np(x):
    N = x.shape[0]
    M = np.array([[complex(1, 0), complex(1, 0)],
                  [complex(1, 0), complex(-1, 0)]], dtype=np.cdouble)
    Ax = np.dot(M, x.reshape((2, N//2)))
    for _ in range(int(np.log2(N)-1)):
        ROWS,COLS_O2 = Ax.shape[0],Ax.shape[1]//2      
        terms = np.exp(-1j*pi*np.arange(ROWS)/ROWS).reshape(ROWS,1)
        X_even = Ax[:,:COLS_O2]
        X_odd = Ax[:,COLS_O2:]
        Ax = np.zeros(shape=(ROWS*2,COLS_O2), dtype=np.cdouble)    
        Ax[0:ROWS] = X_even + terms * X_odd      
        Ax[ROWS:] = X_even - terms * X_odd          
    return Ax.ravel()/N

numba_warm_up = np.array([1,2,3,4,5])
bluestein_np(numba_warm_up)

sr = 100000
t0 = 0
tn = 1
t = np.arange(t0,tn,1/sr)
signal = np.sin(20*2*np.pi*t)+np.cos(200*2*np.pi*t)+np.cos(100*2*np.pi*t+pi/4)
signal_list = signal.tolist()


#%%

res = {}

start = time.time()
fftcpp = blue_cpp(signal_list)
end = time.time()
res["my c++"]=end - start

start = time.time()
fftnp = np.fft.fft(signal)
end = time.time()
res["np.fft.fft"]=end - start


start = time.time()
fftpt = bluestein_np(signal)
end = time.time()
res["my np"]=end - start


start = time.time()
fftli = bluestein_list(signal)
end = time.time()
res["my list"]=end - start



print(f'results: {[False,True][np.sum(np.allclose(np.round(fftnp,2), np.round(np.array(fftcpp.getFourCoeff())),2) *np.allclose(np.round(fftnp,2), np.round(fftpt),2) *np.allclose(np.round(fftnp,2), np.round(np.array(fftli)),2))]}')

# results on data 100000 long:
    # {'my c++': 0.05332493782043457,
    #  'np.fft.fft': 0.009668827056884766,
    #  'my np': 0.06859493255615234,
    #  'my list': 6.298625707626343}   
    
#%%
