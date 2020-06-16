import numpy as np
import sys
import os
from astropy.io import fits
from utilities import Cosmology, Utilities
from scipy.integrate import quad, simps, odeint
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline, interp1d
from scipy.optimize import fsolve
from scipy.signal import savgol_filter
from scipy.special import eval_legendre

class SingleFit:
    def __init__(self,
                 xi_r_filename,
                 xi_smu_filename,
                 covmat_filename=None,
                 sv_filename=None,
                 vr_filename=None,
                 full_fit=1,
                 smin=0,
                 smax=150,
                 model=1,
                 const_sv=0,
                 model_as_truth=0,
                 om_m=0.285,
                 s8=0.828,
                 eff_z=0.57):

        self.xi_r_filename = xi_r_filename
        self.sv_filename = sv_filename
        self.vr_filename = vr_filename
        self.xi_smu_filename = xi_smu_filename
        self.covmat_filename = covmat_filename
        self.smin = smin
        self.smax = smax
        self.const_sv = const_sv
        self.model = model
        self.model_as_truth = model_as_truth

        # full fit (monopole + quadrupole)
        self.full_fit = bool(full_fit)

        print("Setting up redshift-space distortions model.")

        # cosmology for Minerva
        self.om_m = om_m
        self.s8 = s8
        self.cosmo = Cosmology(om_m=self.om_m)
        self.nmocks = 299 # hardcoded for Minerva

        self.eff_z = eff_z
        self.dA = self.cosmo.get_angular_diameter_distance(self.eff_z)

        self.growth = self.cosmo.get_growth(self.eff_z)
        self.f = self.cosmo.get_f(self.eff_z)
        self.b = 2.01
        self.beta = self.f / self.b
        self.s8norm = self.s8 * self.growth 

        eofz = np.sqrt((self.om_m * (1 + self.eff_z) ** 3 + 1 - self.om_m))
        self.iaH = (1 + self.eff_z) / (100. * eofz) 

        # read real-space galaxy monopole
        data = np.genfromtxt(self.xi_r_filename)
        self.r_for_xi = data[:,0]
        xi_r = data[:,1]
        self.xi_r = InterpolatedUnivariateSpline(self.r_for_xi, xi_r, k=3, ext=3)

        # data = np.genfromtxt(self.int_xi_r_filename)
        # self.r_for_xi = data[:,0]
        # int_xi_r = data[:,1]
        # self.int_xi_r = InterpolatedUnivariateSpline(self.r_for_xi, int_xi_r, k=3, ext=3)

        int_xi_r = np.zeros_like(self.r_for_xi)
        dr = np.diff(self.r_for_xi)[0]
        for i in range(len(int_xi_r)):
            int_xi_r[i] = 1./(self.r_for_xi[i]+dr/2)**3 * (np.sum(xi_r[:i+1]*((self.r_for_xi[:i+1]+dr/2)**3
                                                        - (self.r_for_xi[:i+1] - dr/2)**3)))
        self.int_xi_r = InterpolatedUnivariateSpline(self.r_for_xi, int_xi_r, k=3, ext=3)

        # integral = np.zeros_like(self.r_for_xi)
        # for i in range(len(integral)):
        #     integral[i] = quad(lambda x: self.xi_r(x) * x ** 2, 0, self.r_for_xi[i], full_output=1)[0]
        # int_xi_r = 3 * integral / self.r_for_xi ** 3
        # self.int_xi_r = InterpolatedUnivariateSpline(self.r_for_xi, int_xi_r, k=3, ext=3)

        # integral = np.zeros_like(self.r_for_xi)
        # for i in range(len(integral)):
        #     integral[i] = quad(lambda x: self.xi_r(x) * x ** 4, 0, self.r_for_xi[i], full_output=1)[0]
        # int2_xi_r = 5 * integral / self.r_for_xi ** 5
        # self.int2_xi_r = InterpolatedUnivariateSpline(self.r_for_xi, int2_xi_r, k=3, ext=3)

        int2_xi_r = np.zeros_like(self.r_for_xi)
        dr = np.diff(self.r_for_xi)[0]
        for i in range(len(int2_xi_r)):
            int2_xi_r[i] = 1./(self.r_for_xi[i]+dr/2)**5 * (np.sum(xi_r[:i+1]*((self.r_for_xi[:i+1]+dr/2)**5
                                                        - (self.r_for_xi[:i+1] - dr/2)**5)))

        self.int2_xi_r = InterpolatedUnivariateSpline(self.r_for_xi, int2_xi_r, k=3, ext=3)

        if self.model == 1:
            # read los velocity dispersion profile
            data = np.genfromtxt(self.sv_filename)
            self.r_for_sv = data[:,0]
            sv = data[:,-2]
            if self.const_sv:
                sv = np.ones(len(self.r_for_sv))
            else:
                self.sv_converge = sv[-1]
                sv = sv / self.sv_converge
                sv = savgol_filter(sv, 3, 1)
            self.sv = InterpolatedUnivariateSpline(self.r_for_sv, sv, k=3, ext=3)

        # if self.model == 1:
        #     # read radial velocity profile
        #     data = np.genfromtxt(self.vr_filename)
        #     self.r_for_vr = data[:,0]
        #     vr = data[:,-2]
        #     dvr = np.gradient(vr, self.r_for_vr)
        #     self.vr = InterpolatedUnivariateSpline(self.r_for_vr, vr, k=3, ext=3)
        #     self.dvr = InterpolatedUnivariateSpline(self.r_for_vr, dvr, k=3, ext=3)

        # read redshift-space correlation function
        self.s_for_xi, self.mu_for_xi, self.xi_smu = Utilities.readCorrFile(self.xi_smu_filename)

        # if self.model_as_truth:
        #     print('Using the model prediction as the measurement.')
        #     if self.model == 1:
        #         fs8 = self.f * self.s8norm
        #         sigma_v = self.sv_converge
        #         alpha = 1.0
        #         epsilon = 1.0
        #         alpha_para = alpha * epsilon ** (-2/3)
        #         alpha_perp = epsilon * alpha_para

        #         self.xi0_s, self.xi2_s = self.model1_theory(fs8,
        #                                                     sigma_v,
        #                                                     alpha_perp,
        #                                                     alpha_para,
        #                                                     self.s_for_xi,
        #                                                     self.mu_for_xi)
        # else:
        s, self.xi0_s = Utilities.getMonopole(self.s_for_xi, self.mu_for_xi, self.xi_smu)
        s, self.xi2_s = Utilities.getQuadrupole(self.s_for_xi, self.mu_for_xi, self.xi_smu)

        # # read covariance matrix
        # if os.path.isfile(self.covmat_filename):
        #     print('Reading covariance matrix: ' + self.covmat_filename)
        #     self.cov = np.load(self.covmat_filename)
        #     self.icov = np.linalg.inv(self.cov)
        # else:
        #     sys.exit('Covariance matrix not found.')


        # # restrict measured vectors to the desired fitting scales
        # if (self.smax < self.s_for_xi.max()) or (self.smin > self.s_for_xi.min()):

        #     scales = (self.s_for_xi >= self.smin) & (self.s_for_xi <= self.smax)

        #     # truncate redshift-space data vectors
        #     self.s_for_xi = self.s_for_xi[scales]
        #     self.xi0_s = self.xi0_s[scales]
        #     self.xi2_s = self.xi2_s[scales]

        # # build data vector
        # if self.full_fit:
        #     self.datavec = np.concatenate((self.xi0_s, self.xi2_s))
        # else:
        #     self.datavec = self.xi2_s
        
    def model1_theory(self, beta, sigma_v, alpha_perp, alpha_para, s, mu):
        '''
        Gaussian streaming model
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 100)
        r = self.r_for_xi
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y3 = self.int_xi_r(r)
        y4 = self.int2_xi_r(r)
        y5 = self.sv(r)
        

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_int_xi_r = InterpolatedUnivariateSpline(x, y3, k=3, ext=3)
        rescaled_int2_xi_r = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)
        rescaled_sv = InterpolatedUnivariateSpline(x, y5, k=3, ext=3)
        sigma_v = alpha_para * sigma_v

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                rpar = true_spar

                vpar = -1/3 * beta * 1/self.iaH * true_s * rescaled_int_xi_r(true_s) * true_mu[j]

                sy_central = sigma_v * rescaled_sv(true_s)
                y = np.linspace(-5 * sy_central, 5 * sy_central, 100)

                vpary = vpar + y
                rpary = rpar + vpary * self.iaH

                r = np.sqrt(true_sperp**2 + rpary**2)
                sy = sigma_v * rescaled_sv(r)
                v = -1/3 * beta * 1/self.iaH * r * rescaled_int_xi_r(r)

                integrand = (1 + rescaled_xi_r(r)) * np.exp(-0.5 * ((vpary - v*true_mu[j]) / sy)**2) / (np.sqrt(2 * np.pi) * sy)

                xi_model[j] = np.trapz(integrand, vpary) - 1


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                  xi_model[np.argsort(true_mu)],
                                                  k=3)

            if true_mu.min() < 0:
                mumin = -1
                factor = 2
            else:
                mumin = 0
                factor = 1

            # get multipoles
            xaxis = np.linspace(mumin, 1, 1000)

            yaxis = mufunc(xaxis) / factor
            monopole[i] = np.trapz(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / factor
            quadrupole[i] = np.trapz(yaxis, xaxis)
            
        return monopole, quadrupole



    def model2_theory(self, beta, alpha_perp, alpha_para, s, mu):
        '''
        Linear model (Eq. 44-46 from Hernandez-Aguayo et at. 2018)
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 100)
        r = self.r_for_xi
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y3 = self.int_xi_r(r)
        y4 = self.int2_xi_r(r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_int_xi_r = InterpolatedUnivariateSpline(x, y3, k=3, ext=3)
        rescaled_int2_xi_r = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                r = true_s

                xi_model[j] = eval_legendre(0, true_mu[j]) * (1 + 2/3*beta + 1/5 * beta**2) * rescaled_xi_r(r) \
                            + eval_legendre(2, true_mu[j]) * (4/3 * beta + 4/7 * beta**2) * (rescaled_xi_r(r) - rescaled_int_xi_r(r)) \
                            + eval_legendre(4, true_mu[j]) * (8/35 * beta**2) * (rescaled_xi_r(r) + 5/2 * rescaled_int_xi_r(r) - 7/2 * rescaled_int2_xi_r(r))

            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                  xi_model[np.argsort(true_mu)],
                                                  k=3)

            if true_mu.min() < 0:
                mumin = -1
                factor = 2
            else:
                mumin = 0
                factor = 1


            # get multipoles
            xaxis = np.linspace(mumin, 1, 1000)

            yaxis = mufunc(xaxis) / factor
            monopole[i] = np.trapz(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / factor
            quadrupole[i] = np.trapz(yaxis, xaxis)
            
        return monopole, quadrupole

    def model3_theory(self, beta, sigma_v, alpha_perp, alpha_para, s, mu):
        '''
        RSD model from Nadathur & Percival (2018).
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 100)
        r = self.r_for_xi
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y3 = self.int_xi_r(r)
        y4 = self.int2_xi_r(r)
        y5 = self.sv(r)
        

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_int_xi_r = InterpolatedUnivariateSpline(x, y3, k=3, ext=3)
        rescaled_int2_xi_r = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)
        rescaled_sv = InterpolatedUnivariateSpline(x, y5, k=3, ext=3)
        sigma_v = alpha_para * sigma_v

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                rpar = true_spar + true_s * beta * rescaled_int_xi_r(true_s) * true_mu[j] / 3.
                sy_central = sigma_v * rescaled_sv(np.sqrt(true_sperp**2 + rpar**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5 * sy_central, 100)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                sy = sigma_v * rescaled_sv(rr) * self.iaH

                integrand = (1 + rescaled_xi_r(rr)) * \
                            (1 + (beta * rescaled_int_xi_r(rr) / 3. - y * true_mu[j] / rr) * (1 - true_mu[j]**2) +
                             beta * (rescaled_xi_r(rr) - 2 * rescaled_int_xi_r(rr) / 3.) * true_mu[j]**2)
                integrand = integrand * np.exp(-(y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)
                xi_model[j] = np.trapz(integrand, y) - 1


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)], xi_model[np.argsort(true_mu)], k=3)

            if true_mu.min() < 0:
                mumin = -1
                factor = 2
            else:
                mumin = 0
                factor = 1

            # get multipoles
            xaxis = np.linspace(mumin, 1, 1000)

            yaxis = mufunc(xaxis) / factor
            monopole[i] = np.trapz(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / factor
            quadrupole[i] = np.trapz(yaxis, xaxis)

            
        return monopole, quadrupole

    



