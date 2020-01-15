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
md.log.RunScript('vv.Dataset.OpenMap("Evaluation")')

X = md.log.LoadTable()
augX = X[:,0].astype(np.int32)
X = X[:, 1:]
varAugX = md.GetVariable('AugX')
varRep = md.GetVar('AugRep:0')
cR = md.GetTensor(varRep)
R = np.copy(cR)
K = R.shape[0]
N = X.shape[0]//K
#md.log.ShowMatrix(R, view=4)

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

for repeats in range(2):
    if repeats != 0: time.sleep(2.0)
    for k in range(K):        
        for g in np.linspace(0, 1.0, 25):
            R[k,:] = (1-g)*cR[k, :] + g*cR[(k+1)%K,:]
            ShowMap(k)
        R[:,:] = cR[:,:]
        ShowMap(k)
