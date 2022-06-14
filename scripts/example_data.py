import numpy as np
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import sys, os, time, argparse
import flowpm
print(flowpm, "\n")

sys.path.append('../src/')
# sys.path.append('../../galference/utils/')
# sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')
# sys.path.append('../../hmc/src/')

from pmfuncs import Evolve
from pyhmc import PyHMC, PyHMC_batch, DualAveragingStepSize
#from callback import callback_sampling, datafig, corner
#import recon
from flowpm.scipy.interpolate import interp_tf
#import trenfmodel


#$#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--bs', type=float, default=200., help='box size')
parser.add_argument('--nc', type=int, default=32, help='mesh size')
parser.add_argument('--suffix', type=str, help='suffix to fpath')
parser.add_argument('--nsamples', type=int, default=5000, help="number of samples to generate")
parser.add_argument('--debug', type=int, default=0, help="debug run")
parser.add_argument('--dnoise', type=float, default=1., help='noise level, 1 is shot noise')
parser.add_argument('--order', type=int, default=1, help="ZA or LPT")



args = parser.parse_args()

##########
suffix = args.suffix
bs, nc = args.bs, args.nc
nsteps = 3
a0, af, nsteps = 0.1, 1.0,  nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
donbody = False
order = args.order
shotnoise = bs**3/nc**3
dnoise = args.dnoise



# Compute necessary Fourier kernels                                                              
evolve = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)

cosmodata = evolve.cosmodict
params = tf.stack([cosmodata['Omega_c'], cosmodata['sigma8'], cosmodata['Omega_b'], cosmodata['h']])
print("params : ", params)
ndim = params.numpy().size


##################
##Generate DATA

np.random.seed(100)
zic = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc)
ic = evolve.z_to_lin(zic).numpy()
fin = evolve.pm(tf.constant(ic)).numpy()
data = fin + noise
data = data.astype(np.float32)
print("data shape : ", data.shape)
tfdata = tf.constant(data)
tfnoise = tf.constant(dnoise)


##############################################

cosmology = flowpm.cosmology.Planck15(Omega_c=params[0], sigma8=params[1], h=params[2], Omega_b=params[3])
k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
pk = flowpm.tfpower.linear_matter_power(cosmology, k)
pk_fun = lambda x: tf.cast(
    tf.reshape(interp_tf(
            tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.
    complex64)


@tf.function
def whitenoise_to_linear(evolve, white, pk):
    nc, bs = evolve.nc, evolve.bs
    pkmesh = pk(evolve.kmesh)
    whitec = flowpm.utils.r2c3d(white* nc**1.5)
    lineark = tf.multiply(whitec, tf.cast((pkmesh /bs**3)**0.5, whitec.dtype))
    linear = flowpm.utils.c2r3d(lineark, norm=nc**3)
    return linear


@tf.function
def cosmo_sim_fixed(white, retic=False):
    ic = whitenoise_to_linear(evolve, white, pk_fun)
    if retic: return ic
    final_field = evolve.pm(ic, cosmology.to_dict())
    return final_field


@tf.function
def cosmo_sim(params, white, retic=False):
    cosmology = flowpm.cosmology.Planck15(Omega_c=params[0], sigma8=params[1], h=params[2], Omega_b=params[3])
    k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
    pk = flowpm.tfpower.linear_matter_power(cosmology, k)
    pk_fun = lambda x: tf.cast(
        tf.reshape(interp_tf(
                tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.
        complex64)
    ic = whitenoise_to_linear(evolve, white, pk_fun)
    if retic: return ic
    final_field = evolve.pm(ic, cosmology.to_dict())
    return final_field

#params0 = (params.numpy()*np.random.uniform(0.9, 1.1)).astype(np.float32)
#zicmix = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
#white0 = 0.5*(zic + zicmix) #(zic*np.random.uniform(0.9, 1.1, zic.size).reshape(zic.shape)).astype(np.float32)

cosmo_sim(params, zic)
cosmo_sim_fixed(zic)

