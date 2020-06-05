import numpy as np
import sys
import glob
import click
from scipy.integrate import quad, simps
from scipy.interpolate import InterpolatedUnivariateSpline


def readCorrFile(fname):
    data = np.genfromtxt(fname)
    s = np.unique(data[:,0])
    mu = np.unique(data[:,1])

    xi_smu = np.zeros([len(s), len(mu)])
    counter = 0
    for i in range(len(mu)):
        for j in range(len(s)):
            xi_smu[j, i] = data[counter, -1]
            counter += 1

    return s, mu, xi_smu

def getMonopole(s, mu, xi_smu):
        monopole = np.zeros(xi_smu.shape[0])
        for i in range(xi_smu.shape[0]):
            mufunc = InterpolatedUnivariateSpline(mu, xi_smu[i, :], k=3)
            xaxis = np.linspace(-1, 1, 1000)
            yaxis = mufunc(xaxis) / 2
            monopole[i] = simps(yaxis, xaxis)
        return s, monopole

def getQuadrupole(s, mu, xi_smu):
    quadrupole = np.zeros(xi_smu.shape[0])
    for i in range(xi_smu.shape[0]):
        mufunc = InterpolatedUnivariateSpline(mu, xi_smu[i, :], k=3)
        xaxis = np.linspace(-1, 1, 1000)
        yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / 2
        quadrupole[i] = simps(yaxis, xaxis)

    return s, quadrupole

def getCovarianceMatrix(data, norm=False):
    """
    Assumes rows are observations,
    columns are variables
    """
    nobs, nbins = np.shape(data)
    mean = np.mean(data, axis=0)
    cov = np.zeros([nbins, nbins])

    for k in range(nobs):
        for i in range(nbins):
            for j in range(nbins):
                cov[i, j] += (data[k, i] - mean[i])*(data[k, j] - mean[j])

    cov /= nobs - 1
    
    if norm:
        corr = np.zeros_like(cov)
        for i in range(nbins):
            for j in range(nbins):
                corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
        return corr
    else:
        return cov

def getCrossCovarianceMatrix(data1, data2, norm=False):
    """
    Assumes rows are observations,
    columns are variables
    """
    nobs, nbins = np.shape(data1)
    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)
    cov = np.zeros([nbins, nbins])

    for k in range(nobs):
        for i in range(nbins):
            for j in range(nbins):
                cov[i, j] += (data1[k, i] - mean1[i])*(data2[k, j] - mean2[j])

    cov /= nobs - 1
    
    if norm:
        corr = np.zeros_like(cov)
        for i in range(nbins):
            for j in range(nbins):
                corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
        return corr
    else:
        return cov

def MultipoleCovariance(handle_mocks, smin, smax, full_fit=True):
        files_mocks = sorted(glob.glob(handle_mocks))
        mock_datavec = []
        for fname in files_mocks:
            s, mu, xi_smu_mock = readCorrFile(fname)
            s, xi0 = getMonopole(s, mu, xi_smu_mock)
            s, xi2 = getQuadrupole(s, mu, xi_smu_mock)

            # only keep scales to fit later
            idx = (s >= smin) & (s <= smax)
            s = s[idx]
            xi0 = xi0[idx]
            xi2 = xi2[idx]

            if full_fit:
                datavec = np.concatenate((xi0, xi2))
            else:
                datavec = xi2

            mock_datavec.append(datavec)

        mock_datavec = np.asarray(mock_datavec)
        cov = getCovarianceMatrix(mock_datavec)
        return cov

@click.command()
@click.option('--handle_in', type=str, required=True, help='Handle from mocks')
@click.option('--handle_out', type=str, required=True, help='Handle for the mean')
@click.option('--smin', type=float, default=0.0, help='Minimum scale to fit (in Mpc/h)')
@click.option('--smax', type=float, default=100.0, help='Maximum scale to fit (in Mpc/h)')
@click.option('--full_fit', type=bool, default=True, help='Full fit or quadrupole only?')

def get_covariance(handle_in,
                   handle_out,
                   smin,
                   smax,
                   full_fit):

    cov = MultipoleCovariance(handle_mocks=handle_in,
                              smin=smin,
                              smax=smax,
                              full_fit=full_fit)

    np.save(handle_out, cov)

if __name__ == '__main__':
    get_covariance()