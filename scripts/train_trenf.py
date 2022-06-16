import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions



import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--jobid', type=int, default=1, help='an integer for the accumulator')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--nc', type=int, default=32, help='mesh size')
parser.add_argument('--bs', type=float, default=200, help='box size')
parser.add_argument('--suffix', type=str, default="", help='suffix to fpath')
parser.add_argument('--nR', type=int, default=0, help="number of smoothings")
parser.add_argument('--nlayers', type=int, default=3, help="number of trenf layers")
parser.add_argument('--nbins', type=int, default=32, help="number of bins in trenf spline")
parser.add_argument('--mode', type=str, default="classic", help='')
parser.add_argument('--nknots', type=int, default=100, help='number of trenf layers')
parser.add_argument('--linknots', type=int, default=0, help='linear spacing for knots')
parser.add_argument('--kwts', type=int, default=4, help='number of trenf layers')
parser.add_argument('--fitnoise', type=int, default=0, help='fitnoise')
parser.add_argument('--fitscale', type=int, default=1, help='fitscale')
parser.add_argument('--fitmean', type=int, default=0, help='fitmean')
parser.add_argument('--meanfield', type=int, default=1, help='meanfield for affine')
parser.add_argument('--ntrain', type=int, default=2000, help='number of training iterations')
parser.add_argument('--regwt0', type=float, default=0., help='regularization weight')

args = parser.parse_args()
device = args.jobid
suffix = args.suffix

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#print("\nDevice name\n", tf.test.gpu_device_name(), "\n")
print("\nDevices\n", get_available_gpus())


from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import scipy.optimize as sopt
import sys, os, flowpm

#sys.path.append('../src/')
import trenfmodel
from pmfuncs import Evolve
#from pyhmc import PyHMC, PyHMC_batch
#from callback import callback, datafig, callback_fvi, callback_sampling
#import tools
#import diagnostics as dg

##########
bs, nc = args.bs, args.nc
nsteps = 3
a0, af = 0.1, 1.0
stages = np.linspace(a0, af, nsteps, endpoint=True)
donbody = False
order = 1
evolve = Evolve(nc, bs,  a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)     


##############################################

print("\nFor seed : ", args.seed)
np.random.seed(args.seed)

@tf.function
def train_function(model, samples, opt):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logq = - tf.reduce_mean(model.q.log_prob(samples))
    gradients = tape.gradient(logq, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return logq



###################################################################################

trenf = trenfmodel.TRENF(nc, nlayers=args.nlayers, evolve=evolve, nbins=args.nbins, nknots=args.nknots, mode=args.mode, \
                             linknots=bool(args.linknots), fitnoise=bool(args.fitnoise), fitscale=bool(args.fitscale), \
                             fitmean=bool(args.fitmean), meanfield=bool(args.meanfield))
offset = tf.Variable(0.)*0.

for i in trenf.variables:
    print(i.name, i.shape)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-2,
    decay_steps=100,
    decay_rate=0.8,
    staircase=True)

#Train FLOW using previos samples assuming we have them
samples = np.squeeze(np.array([np.load('./tmp/%d.npy'%i) for i in range(5)]))
print(samples.shape)

trainsamples = samples[:4].copy()
testsamples = samples[4:].copy()
print("Test train split : ", trainsamples.shape, testsamples.shape)

ntrain = 500
nsamples = 2
losses = []
opt = tf.keras.optimizers.Adam(learning_rate= lr_schedule)
saveiter = 1000

#generate a sample before training
plt.imshow(trenf.sample(1)[0].numpy().sum(axis=0))
plt.colorbar()
plt.savefig('./tmp/initsample')
plt.colorbar()
plt.close()

print("Start training")
for i in range(ntrain):
    idx = np.random.choice(trainsamples.shape[0], nsamples)
    batch = tf.concat(trainsamples[idx], axis=0)
    losses.append(train_function(trenf, batch, opt))
    #save weights
#     if (i > 0) & (i%saveiter == 0): 
#         trenf.save_weights(fpath + '/weights/iter%04d'%(i//saveiter))
print("Training finished")

plt.figure()
plt.plot(losses)
plt.savefig('./tmp/losses.png')

#generate a sample after training
plt.figure()
plt.imshow(trenf.sample(1)[0].numpy().sum(axis=0))
plt.colorbar()
plt.savefig('./tmp/finsample')
plt.colorbar()
plt.close()

