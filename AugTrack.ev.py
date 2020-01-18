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

X = md.log.LoadTable()
augX = X[:,0].astype(np.int32)
X = X[:, 1:]
varAugX = md.GetVariable('AugX')
varRep = md.GetVar('AugRep:0')
cR = md.GetTensor(varRep)
R = np.copy(cR)
K = R.shape[0]
N = X.shape[0]//K
map = None
mapSize = 1227.0


def ShowInitMap():
    global map
    R[:,:] = cR[:,:]
    varRep.load(R, md.sess)
    map = md.sess.run(md.Output(), {md.inputHod:X, varAugX:augX})
    md.log.ShowMatrix(mapSize*map, view=13, access='r')

def ShowMap(k):
    global map
    varRep.load(R, md.sess)
    rr = slice(k*N, k*N+N)
    map[rr, :] = md.sess.run(md.Output(), {md.inputHod:X[rr, :], varAugX:augX[rr]})
    md.log.ShowMatrix(mapSize*map, view=13, access='r')
    #time.sleep(0.01)

def Loop(repeats, steps):
    for _ in range(repeats):
        ShowInitMap()
        if repeats != 0: time.sleep(1.0)
        for k in range(K):        
            for g in np.linspace(0, 1.0, steps):
                R[k,:] = (1-g)*cR[k, :] + g*cR[(k+1)%K,:]
                ShowMap(k)        

Loop(1, 12)
info = np.arange(K) + (1<<16)
md.log.ShowMatrix(R, view=2, access='r', rowInfo=info)