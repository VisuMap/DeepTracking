#=========================================================================
# Model generator for regression.
#=========================================================================
import ModelUtil as mu
import numpy as np
import tensorflow as tf

co = mu.CmdOptions()
ds = mu.ModelDataset('+', 'Shp')
augX = ds.X[:,0].astype(np.int32)
ds.X = ds.X[:,1:]
ds.UpdateAux()

md = mu.ModelBuilder(ds.xDim, ds.yDim, job=co.job)
netCfg = 5 * [30]
md.r0 = 0.0005
augDim = 3
augLen = np.max(augX) + 1

md.AddLayers(netCfg[0])
md.AddDropout()
md.AddLayers(netCfg[1:-1])
varAugX, _ = md.AddAugmentIndexed(augLen, augDim, binding=1)
md.AddLayers(netCfg[-1:])
md.AddScalingTo(ds.Y)
md.cost = md.SquaredCost(md.Output(), md.Label()) 
md.SetAdamOptimizer(co.epochs, ds.N)

idxMap = np.arange(ds.N)
np.random.shuffle(idxMap)
_X, _augX = ds.X, augX
ds.X = np.take(ds.X, idxMap, axis=0)
ds.Y = np.take(ds.Y, idxMap, axis=0)
augX = np.take(augX, idxMap, axis=0)

class AugInterp:
    idx = 0
    def InitEpoch(self, md):
        self.idx = 0
    def BeginStep(self):
        if self.idx >= ds.N:
            return False
        rr = slice(self.idx, self.idx+md.batchSize)
        md.feed[md.inputHod] = ds.X[rr]
        md.feed[varAugX] = augX[rr]
        md.feed[md.outputHod] = ds.Y[rr]
        self.idx = rr.stop        
        return True

def Monitor(ep):
    print('%d: %.6g'%(ep, md.lastError))
    md.log.ReportCost(ep, md.lastError, md.job)
    map  = md.sess.run(md.Output(), {md.inputHod:_X, varAugX:_augX})
    md.ShowTensorMap(map)
    if ep == co.epochs//2:
        md.SetVar(md.keepProbVar, 1.0)

md.Train(AugInterp(), co.epochs, co.logLevel, co.refreshFreq, epCall=Monitor)
md.Save(co.modelName)
