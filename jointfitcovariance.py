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

def MultipoleCovariance(handle_mocks, smins, smaxs, full_fit=True):
        handle_mocks = handle_mocks.split(',')
        smins = [int(i) for i in smins.split(',')]
        smaxs = [int(i) for i in smaxs.split(',')]
        ndenbins = len(handle_mocks)

        files_mocks = {}
        smin = {}
        smax = {}
        for j in range(ndenbins):
            denbin = 'den{}'.format(j)
            files_mocks[denbin] = sorted(glob.glob(handle_mocks[j]))
            smin[denbin] = smins[j]
            smax[denbin] = smaxs[j]

        nmocks = len(files_mocks['den0'])
        mock_datavec = []

        for i in range(nmocks):
            datavec = np.array([])
            for j in range(ndenbins):
                denbin = 'den{}'.format(j)
                fname = files_mocks[denbin][i]
                s, mu, xi_smu_mock = readCorrFile(fname)
                s, xi0 = getMonopole(s, mu, xi_smu_mock)
                s, xi2 = getQuadrupole(s, mu, xi_smu_mock)

                # only keep scales to fit later
                fitscale = (s >= smin[denbin]) & (s <= smax[denbin])
                s = s[fitscale]
                xi0 = xi0[fitscale]
                xi2 = xi2[fitscale]

                if full_fit:
                    datavec = np.concatenate((datavec, xi0, xi2))
                else:
                    datavec = np.concatenate((datavec, xi2))

            mock_datavec.append(datavec)


        mock_datavec = np.asarray(mock_datavec)
        cov = getCovarianceMatrix(mock_datavec)
        return cov

@click.command()
@click.option('--handle_in', type=str, required=True, help='Handle from mocks')
@click.option('--handle_out', type=str, required=True, help='Handle for the mean')
@click.option('--smins', type=str, default=0.0, help='Minimum scale to fit (in Mpc/h)')
@click.option('--smaxs', type=str, default=100.0, help='Maximum scale to fit (in Mpc/h)')
@click.option('--full_fit', type=bool, default=True, help='Full fit or quadrupole only?')

def get_covariance(handle_in,
                   handle_out,
                   smins,
                   smaxs,
                   full_fit):

    cov = MultipoleCovariance(handle_mocks=handle_in,
                              smins=smins,
                              smaxs=smaxs,
                              full_fit=full_fit)

    np.save(handle_out, cov)

if __name__ == '__main__':
    get_covariance()