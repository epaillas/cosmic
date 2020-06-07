import numpy as np
from rsdmodel import JointFit
import os
import sys
import argparse
import emcee
from multiprocessing import Pool

def log_probability(theta):
        lp = model.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + model.log_likelihood(theta)

parser = argparse.ArgumentParser(description='')

parser.add_argument('--ncores', type=int)
parser.add_argument('--xi_smu', type=str)
parser.add_argument('--xi_r', type=str)
parser.add_argument('--delta_r', type=str)
parser.add_argument('--int_delta_r', type=str)
parser.add_argument('--sv_r', type=str)
parser.add_argument('--covmat', type=str)
parser.add_argument('--full_fit', type=int, default=1)
parser.add_argument('--smin', type=str)
parser.add_argument('--smax', type=str)
parser.add_argument('--model', type=int, default=1)
parser.add_argument('--const_sv', type=int, default=0)
parser.add_argument('--model_as_truth', type=int, default=0)
parser.add_argument('--backend_name', type=str)
parser.add_argument('--ndenbins', type=int)

args = parser.parse_args()  

os.environ["OMP_NUM_THREADS"] = "1"

if args.model == 1:
    model = JointFit(delta_r_filenames=args.delta_r,
                   int_delta_r_filenames=args.int_delta_r,
                   xi_r_filenames=args.xi_r,
                   sv_filenames=args.sv_r,
                   xi_smu_filenames=args.xi_smu,
                   covmat_filename=args.covmat,
                   full_fit=args.full_fit,
                   smins=args.smin,
                   smaxs=args.smax,
                   model=args.model,
                   const_sv=args.const_sv,
                   model_as_truth=args.model_as_truth)

    nwalkers = args.ncores
    niter = 10000

    fs8 = 0.472
    epsilon = 1.0

    if args.ndenbins == 2:
        ndim = 4
        sigma_v1 = 360
        sigma_v2 = 360
        start_params = np.array([fs8, sigma_v1, sigma_v2, epsilon])
        scales = [1, 100, 100, 1]

    if args.ndenbins == 3:
        ndim = 5
        sigma_v1 = 360
        sigma_v2 = 360
        sigma_v3 = 360
        start_params = np.array([fs8, sigma_v1, sigma_v2, sigma_v3, epsilon])
        scales = [1, 100, 100, 100, 1]

    if args.ndenbins == 5:
        ndim = 7
        sigma_v1 = 360
        sigma_v2 = 360
        sigma_v3 = 360
        sigma_v4 = 360
        sigma_v5 = 360
        start_params = np.array([fs8, sigma_v1, sigma_v2, sigma_v3, sigma_v4, sigma_v5, epsilon])
        scales = [1, 100, 100, 100, 100, 100, 1]

    p0 = [start_params + 1e-2 * np.random.randn(ndim) * scales for i in range(nwalkers)]

    print('Running emcee with the following parameters:')
    print('nwalkers: ' + str(nwalkers))
    print('ndim: ' + str(ndim))
    print('niter: ' + str(niter))
    print('backend: ' + args.backend_name)
    print('Running in {} CPUs'.format(args.ncores))

    backend = emcee.backends.HDFBackend(args.backend_name)
    backend.reset(nwalkers, ndim)

    with Pool(processes=args.ncores) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                        log_probability,
                                        backend=backend,
                                        pool=pool)
                                        
        sampler.run_mcmc(p0, niter, progress=True)






