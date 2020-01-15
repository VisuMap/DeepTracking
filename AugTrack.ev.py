#=========================================================================
# Eval.ev.py
#=========================================================================
import sys,os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
from ModelUtil import ModelBuilder

mdName = sys.argv[1]
md = ModelBuilder()
md.LoadModel(mdName)

X = md.log.LoadTable()[:,1:]
K = 4
N = X.shape[0]//K
varAugX, augX = md.GetVariable('augX'), np.zeros([K*N], dtype=np.int32)
for k in range(K):
    augX[k*N:k*N+N] = k
varRep = md.GetVar('AugRep:0')
cR = md.GetTensor(varRep)
R = np.copy(cR)
md.log.ShowMatrix(R, view=4)

map = md.sess.run(md.Output(), {md.inputHod:X, varAugX:augX})
md.log.AppMapping2(map)
time.sleep(2.0)

def ShowMap(k):
    global map
    varRep.load(R, md.sess)
    rr = slice(k*N, k*N+N)
    map[rr, :] = md.sess.run(md.Output(), {md.inputHod:X[rr, :], varAugX:augX[rr]})
    md.log.AppMapping2(map)
    time.sleep(0.1)

for reapts in range(1):
    for k in range(K):        
        for g in np.linspace(0, 1.0, 25):
            R[k,:] = (1-g)*cR[k, :] + g*cR[(k+1)%K,:]
            ShowMap(k)
        R[:,:] = cR[:,:]
        ShowMap(k)
    time.sleep(2.0)
