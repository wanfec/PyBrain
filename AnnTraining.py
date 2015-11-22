#!/usr/bin/env python

import scipy.io as sio
from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import numpy
import pickle
import arac
import cPickle
from pybrain.structure import SigmoidLayer
from numpy import sort

fo01  = open('out', 'w')
fo02  = open('dist', 'w')
fonet = open('net', 'w')

#load mat from matlab
mat = sio.loadmat('Features.mat')
#print(mat)
X = mat['X']
y = mat['y']
length = X.shape[0]

#set data
alldata = SupervisedDataSet(14, 7)
for n in arange(0, length):
    alldata.appendLinked(X[n], y[n])
    
#split data into test data and training data
tstdata, trndata = alldata.splitWithProportion( 0.25 )

#build network
fnn = buildNetwork( trndata.indim, 100, trndata.outdim,  outclass=SigmoidLayer, fast=True)
#print fnn

#build trainer
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.0, verbose=True, weightdecay=0.0)

#start training
trainer.trainUntilConvergence(maxEpochs=1)
#trainer.trainEpochs( 5 )

#save the network using cpickle
cPickle.dump(fnn, fonet)
fonet.close()
out = fnn.activateOnDataset(tstdata)

#print out
print type(out)
out2 = (out>0.5)
print out2
print tstdata['target']
right_number = (out2  == tstdata['target'])
print right_number.sum()
print out2.size
print right_number.sum()/float(out2.size)
hdistance = ((out2-tstdata['target'])**2).sum(axis=1)
hdistance.sort()
print hdistance
#out.tolist()
#print list(out[1:2])
#fo01.write(str(out.tolist()))
#fo02.write( hdistance)

#save the output
numpy.savetxt(fo02, hdistance)

