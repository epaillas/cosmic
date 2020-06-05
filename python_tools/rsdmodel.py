import numpy as np
import sys
import os
from astropy.io import fits
from utilities import Cosmology, Utilities
from scipy.integrate import quad, simps, odeint
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline, interp1d
from scipy.optimize import fsolve
from scipy.signal import savgol_filter

class SingleFit:
    def __init__(self,
                 delta_r_filename,
                 int_delta_r_filename,
                 xi_r_filename,
                 xi_smu_filename,
                 covmat_filename,
                 sv_filename=None,
                 vr_filename=None,
                 full_fit=1,
                 smin=0,
                 smax=100,
                 model=1,
                 const_sv=0,
                 model_as_truth=0,
                 om_m=0.285,
                 s8=0.828,
                 eff_z=0.57):

        self.delta_r_filename = delta_r_filename
        self.int_delta_r_filename = int_delta_r_filename
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
        self.nmocks = 300 # hardcoded for Minerva

        self.eff_z = eff_z
        self.dA = self.cosmo.get_angular_diameter_distance(self.eff_z)

        self.growth = self.cosmo.get_growth(self.eff_z)
        self.f = self.cosmo.get_f(self.eff_z)
        print(self.f)
        self.s8norm = self.s8 * self.growth 

        eofz = np.sqrt((self.om_m * (1 + self.eff_z) ** 3 + 1 - self.om_m))
        self.iaH = (1 + self.eff_z) / (100. * eofz) 

        # read real-space galaxy monopole
        data = np.genfromtxt(self.xi_r_filename)
        self.r_for_xi = data[:,0]
        xi_r = data[:,-2]
        self.xi_r = InterpolatedUnivariateSpline(self.r_for_xi, xi_r, k=3, ext=3)

        # read void-matter correlation function
        data = np.genfromtxt(self.delta_r_filename)
        self.r_for_delta = data[:,0]
        delta_r = data[:,-2]
        self.delta_r = InterpolatedUnivariateSpline(self.r_for_delta, delta_r, k=3, ext=3)

        # read integrated void-matter correlation function
        data = np.genfromtxt(self.int_delta_r_filename)
        self.r_for_delta = data[:,0]
        Delta_r = data[:,-2]
        self.Delta_r = InterpolatedUnivariateSpline(self.r_for_delta, Delta_r, k=3, ext=3)


        if self.model == 1 or self.model == 3 or self.model == 4:
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

        if self.model == 3 or self.model == 4:
            # read radial velocity profile
            data = np.genfromtxt(self.vr_filename)
            self.r_for_vr = data[:,0]
            vr = data[:,-2]
            dvr = np.gradient(vr, self.r_for_vr)
            self.vr = InterpolatedUnivariateSpline(self.r_for_vr, vr, k=3, ext=3)
            self.dvr = InterpolatedUnivariateSpline(self.r_for_vr, dvr, k=3, ext=3)

        # read redshift-space correlation function
        self.s_for_xi, self.mu_for_xi, xi_smu_obs = Utilities.readCorrFile(self.xi_smu_filename)

        if self.model_as_truth:
            print('Using the model prediction as the measurement.')
            if self.model == 1:
                fs8 = self.f * self.s8norm
                sigma_v = self.sv_converge
                alpha = 1.0
                epsilon = 1.0
                alpha_para = alpha * epsilon ** (-2/3)
                alpha_perp = epsilon * alpha_para

                self.xi0_s, self.xi2_s = self.model1_theory(fs8,
                                                            sigma_v,
                                                            alpha_perp,
                                                            alpha_para,
                                                            self.s_for_xi,
                                                            self.mu_for_xi)
        else:
            s, self.xi0_s = Utilities.getMonopole(self.s_for_xi, self.mu_for_xi, xi_smu_obs)
            s, self.xi2_s = Utilities.getQuadrupole(self.s_for_xi, self.mu_for_xi, xi_smu_obs)

        # read covariance matrix
        if os.path.isfile(self.covmat_filename):
            print('Reading covariance matrix: ' + self.covmat_filename)
            self.cov = np.load(self.covmat_filename)
            self.icov = np.linalg.inv(self.cov)
        else:
            sys.exit('Covariance matrix not found.')


        # restrict measured vectors to the desired fitting scales
        if (self.smax < self.s_for_xi.max()) or (self.smin > self.s_for_xi.min()):

            scales = (self.s_for_xi >= self.smin) & (self.s_for_xi <= self.smax)

            # truncate redshift-space data vectors
            self.s_for_xi = self.s_for_xi[scales]
            self.xi0_s = self.xi0_s[scales]
            self.xi2_s = self.xi2_s[scales]

        # build data vector
        if self.full_fit:
            self.datavec = np.concatenate((self.xi0_s, self.xi2_s))
        else:
            self.datavec = self.xi2_s
        

    def log_likelihood(self, theta):
        if self.model == 1:
            fs8, sigma_v, epsilon = theta
        else:
            fs8, epsilon = theta

        alpha = 1.0
        alpha_para = alpha * epsilon ** (-2/3)
        alpha_perp = epsilon * alpha_para

        if self.model == 1 or self.model == 3:
            xi0, xi2 = self.model1_theory(fs8,
                                          sigma_v,
                                          alpha_perp,
                                          alpha_para,
                                          self.s_for_xi,
                                          self.mu_for_xi)
        else:
            xi0, xi2 = self.model2_theory(fs8,
                                          alpha_perp,
                                          alpha_para,
                                          self.s_for_xi,
                                          self.mu_for_xi)

        if self.full_fit:
            modelvec = np.concatenate((xi0, xi2))
        else:
            modelvec = xi2

        chi2 = np.dot(np.dot((modelvec - self.datavec), self.icov), modelvec - self.datavec)
        loglike = -self.nmocks/2 * np.log(1 + chi2/(self.nmocks-1))
        return loglike

    def log_prior(self, theta):
        if self.model == 1 or self.model == 3:
            fs8, sigma_v, epsilon = theta
            if 0.1 < fs8 < 2.0 and 1 < sigma_v < 500 and 0.8 < epsilon < 1.2:
                return 0.0
        else:
            fs8, epsilon = theta
            if 0.1 < fs8 < 2.0 and 0.8 < epsilon < 1.2:
                return 0.0
        
        return -np.inf


    def model1_theory(self, fs8, sigma_v, alpha_perp, alpha_para, s, mu):
        '''
        RSD model from Nadathur & Percival (2018).
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))
        scaled_fs8 = fs8 / self.s8norm

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 80)
        r = self.r_for_delta
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y2 = self.delta_r(r)
        y3 = self.Delta_r(r)
        y4 = self.sv(r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_delta_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=3)
        rescaled_Delta_r = InterpolatedUnivariateSpline(x, y3, k=3, ext=3)
        rescaled_sv = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)
        sigma_v = alpha_para * sigma_v
 

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                rpar = true_spar + true_s * scaled_fs8 * rescaled_Delta_r(true_s) * true_mu[j] / 3.
                sy_central = sigma_v * rescaled_sv(np.sqrt(true_sperp**2 + rpar**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5 * sy_central, 100)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                sy = sigma_v * rescaled_sv(rr) * self.iaH

                integrand = (1 + rescaled_xi_r(rr)) * \
                            (1 + (scaled_fs8 * rescaled_Delta_r(rr) / 3. - y * true_mu[j] / rr) * (1 - true_mu[j]**2) +
                             scaled_fs8 * (rescaled_delta_r(rr) - 2 * rescaled_Delta_r(rr) / 3.) * true_mu[j]**2)
                integrand = integrand * np.exp(-(y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)
                xi_model[j] = np.trapz(integrand, y) - 1


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)], xi_model[np.argsort(true_mu)], k=3)

            # get multipoles
            xaxis = np.linspace(-1, 1, 1000)

            yaxis = mufunc(xaxis) / 2
            monopole[i] = simps(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / 2
            quadrupole[i] = np.trapz(yaxis, xaxis)

            
        return monopole, quadrupole

    def model2_theory(self, fs8, alpha_perp, alpha_para, s, mu):
        '''
        RSD model from Nadathur & Percival (2018)
        without including velocity dispersion.
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))
        scaled_fs8 = fs8 / self.s8norm

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 100)
        r = self.r_for_delta
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y2 = self.delta_r(r)
        y3 = self.Delta_r(r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_delta_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=3)
        rescaled_Delta_r = InterpolatedUnivariateSpline(x, y3, k=3, ext=3)

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                r = true_s * (1 + scaled_fs8/3 * rescaled_Delta_r(true_s) * true_mu[j]**2)

                xi_model[j] = rescaled_xi_r(r) + scaled_fs8/3 * rescaled_Delta_r(r) *\
                            (1 + rescaled_xi_r(r)) + scaled_fs8 * true_mu[j]**2 *\
                            (rescaled_delta_r(r) - rescaled_Delta_r(r)) * \
                            (1 + rescaled_xi_r(r))


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                  xi_model[np.argsort(true_mu)],
                                                  k=3)

            # get multipoles
            xaxis = np.linspace(-1, 1, 1000)

            yaxis = mufunc(xaxis) / 2
            monopole[i] = simps(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / 2
            quadrupole[i] = simps(yaxis, xaxis)
            
        return monopole, quadrupole

    def model3_theory(self, sigma_v, alpha_perp, alpha_para, s, mu):
        '''
        Exact RSD model that used the measured radial
        velocity profiles as an input. It doesn't 
        constraint fs8.
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 100)
        r = self.r_for_delta
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y4 = self.vr(r)
        y5 = self.dvr(r)
        y6 = self.sv(r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_vr = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)
        rescaled_dvr = InterpolatedUnivariateSpline(x, y5, k=3, ext=3)
        rescaled_sv = InterpolatedUnivariateSpline(x, y6, k=3, ext=3)
        sigma_v = alpha_para * sigma_v
 

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                def residual(rpar):
                    rperp = true_sperp
                    r = np.sqrt(rpar**2 + rperp**2)
                    mu = rpar / r
                    res = rpar - true_spar + rescaled_vr(r)*mu * self.iaH
                    return res

                rpar = fsolve(func=residual, x0=true_spar)[0]

                sy_central = sigma_v * rescaled_sv(np.sqrt(true_sperp**2 + rpar**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5 * sy_central, 100)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                sy = sigma_v * rescaled_sv(rr) * self.iaH


                integrand = (1 + rescaled_xi_r(rr)) * (1 + rescaled_vr(rr)/(rr/self.iaH) +\
                                                  (rescaled_dvr(rr) - rescaled_vr(rr)/rr)*self.iaH * true_mu[j]**2)**(-1)

                integrand = integrand * np.exp(-(y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)

                xi_model[j] = np.trapz(integrand, y) - 1
 


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                  xi_model[np.argsort(true_mu)],
                                                  k=3)
            
            # get multipoles
            xaxis = np.linspace(-1, 1, 1000)

            yaxis = mufunc(xaxis) / 2
            monopole[i] = simps(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / 2
            quadrupole[i] = simps(yaxis, xaxis)
            
            
        return monopole, quadrupole

    def model4_theory(self, om_m0, sigma_v, alpha_perp, alpha_para, s, mu):
        '''
        RSD model that relies on spherical collapse model 
        to map from densities to velocities.
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        om_l0 = 1 - om_m0
        zi = 999
        zf = self.eff_z
        z = np.linspace(zi, zf, 100)
        a = 1/(1 + z)
        t = CosmologicalTime(zi, zf)
        delta_lin = np.linspace(-0.01, 0.0025, 1000)

        sol1 = []
        sol2 = []

        for dl in delta_lin:
            g0 = [1 - dl/3, -dl/3]
            sol = odeint(SphericalCollapse, g0, t, args=(om_m0, om_l0))
            y = sol[:,0]
            yprime = sol[:,1]

            sol1.append(y[-1]**-3 - 1)
            sol2.append(yprime[-1])

        interp_den = InterpolatedUnivariateSpline(sol1, delta_lin, k=3, ext=3)
        interp_vel = InterpolatedUnivariateSpline(delta_lin, sol2, k=3, ext=3)

        matched_dls = interp_den(self.Delta_r(self.r_for_delta))
        matched_vpecs = interp_vel(matched_dls)

        H = Hubble(a=a[-1], om_m0=om_m0, om_l0=om_l0)
        q = self.r_for_delta * a[-1] * (1 + self.Delta_r(self.r_for_delta))**(1/3) 
        vpec = matched_vpecs * H * q
        dvpec = np.gradient(vpec, self.r_for_delta)

        self.vr = InterpolatedUnivariateSpline(self.r_for_delta, vpec, k=3, ext=3)
        self.dvr = InterpolatedUnivariateSpline(self.r_for_delta, dvpec, k=3, ext=3)

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 80)
        r = self.r_for_delta
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y4 = self.vr(r)
        y5 = self.dvr(r)
        y6 = self.sv(r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_vr = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)
        rescaled_dvr = InterpolatedUnivariateSpline(x, y5, k=3, ext=3)
        rescaled_sv = InterpolatedUnivariateSpline(x, y6, k=3, ext=3)
        sigma_v = alpha_para * sigma_v

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                def residual(rpar):
                    rperp = true_sperp
                    r = np.sqrt(rpar**2 + rperp**2)
                    mu = rpar / r
                    res = rpar - true_spar + rescaled_vr(r)*mu * self.iaH
                    return res

                rpar = fsolve(func=residual, x0=true_spar)[0]

                sy_central = sigma_v * rescaled_sv(np.sqrt(true_sperp**2 + rpar**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5 * sy_central, 100)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                sy = sigma_v * rescaled_sv(rr) * self.iaH


                integrand = (1 + rescaled_xi_r(rr)) * (1 + rescaled_vr(rr)/(rr/self.iaH) +\
                                                (rescaled_dvr(rr) - rescaled_vr(rr)/rr)*self.iaH * true_mu[j]**2)**(-1)

                integrand = integrand * np.exp(-(y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)

                xi_model[j] = np.trapz(integrand, y) - 1


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                xi_model[np.argsort(true_mu)],
                                                k=3)
            
            # get multipoles
            xaxis = np.linspace(-1, 1, 1000)

            yaxis = mufunc(xaxis) / 2
            monopole[i] = np.trapz(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / 2
            quadrupole[i] = np.trapz(yaxis, xaxis)
            
            
        return monopole, quadrupole

    def model5_theory(self, fs8, sigma_v, alpha_perp, alpha_para, s, mu):
        '''
        RSD model that relies on the Lam et al. (2018)
        non-linear approximation to map densities to velocities.
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))
        scaled_fs8 = fs8 / self.s8norm
        delta_c = 5 # hard coded for now

        vpec = -1/3 * self.r_for_delta * 1/self.iaH * scaled_fs8 \
                * delta_c * ((1 + self.Delta_r(self.r_for_delta))**(1/delta_c) - 1)

        dvpec = np.gradient(vpec, self.r_for_delta)
        self.vr = InterpolatedUnivariateSpline(self.r_for_delta, vpec, k=3, ext=3)
        self.dvr = InterpolatedUnivariateSpline(self.r_for_delta, dvpec, k=3, ext=3)

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 80)
        r = self.r_for_delta
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y4 = self.vr(r)
        y5 = self.dvr(r)
        y6 = self.sv(r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_vr = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)
        rescaled_dvr = InterpolatedUnivariateSpline(x, y5, k=3, ext=3)
        rescaled_sv = InterpolatedUnivariateSpline(x, y6, k=3, ext=3)
        sigma_v = alpha_para * sigma_v


        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                def residual(rpar):
                    rperp = true_sperp
                    r = np.sqrt(rpar**2 + rperp**2)
                    mu = rpar / r
                    res = rpar - true_spar + rescaled_vr(r)*mu * self.iaH
                    return res

                rpar = fsolve(func=residual, x0=true_spar)[0]

                sy_central = sigma_v * rescaled_sv(np.sqrt(true_sperp**2 + rpar**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5 * sy_central, 100)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                sy = sigma_v * rescaled_sv(rr) * self.iaH


                integrand = (1 + rescaled_xi_r(rr)) * (1 + rescaled_vr(rr)/(rr/self.iaH) +\
                                                (rescaled_dvr(rr) - rescaled_vr(rr)/rr)*self.iaH * true_mu[j]**2)**(-1)

                integrand = integrand * np.exp(-(y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)

                xi_model[j] = np.trapz(integrand, y) - 1



            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                xi_model[np.argsort(true_mu)],
                                                k=3)
            
            # get multipoles
            xaxis = np.linspace(-1, 1, 1000)

            yaxis = mufunc(xaxis) / 2
            monopole[i] = np.trapz(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / 2
            quadrupole[i] = np.trapz(yaxis, xaxis)
            
            
        return monopole, quadrupole



class JointFit:
    def __init__(self,
                 delta_r_filenames,
                 int_delta_r_filenames,
                 xi_r_filenames,
                 xi_smu_filenames,
                 covmat_filename,
                 smins,
                 smaxs,
                 sv_filenames=None,
                 full_fit=1,
                 model=1,
                 model_as_truth=0,
                 const_sv=0,
                 om_m=0.285,
                 s8=0.828,
                 eff_z=0.57):

        delta_r_filenames = delta_r_filenames.split(',')
        int_delta_r_filenames = int_delta_r_filenames.split(',')
        xi_r_filenames = xi_r_filenames.split(',')
        sv_filenames = sv_filenames.split(',')
        xi_smu_filenames = xi_smu_filenames.split(',')
        smins = [int(i) for i in smins.split(',')]
        smaxs = [int(i) for i in smaxs.split(',')]

        self.ndenbins = len(delta_r_filenames)
        delta_r_filename = {}
        int_delta_r_filename  = {}
        xi_r_filename = {}
        sv_filename = {}
        xi_smu_filename = {}
        smin = {}
        smax = {}

        for j in range(self.ndenbins):
            delta_r_filename['den{}'.format(j)] = delta_r_filenames[j]
            int_delta_r_filename['den{}'.format(j)] = int_delta_r_filenames[j]
            xi_r_filename['den{}'.format(j)] = xi_r_filenames[j]
            sv_filename['den{}'.format(j)] = sv_filenames[j]
            xi_smu_filename['den{}'.format(j)] = xi_smu_filenames[j]
            smin['den{}'.format(j)] = smins[j]
            smax['den{}'.format(j)] = smaxs[j]
            

        # full fit (monopole + quadrupole)
        self.full_fit = full_fit
        self.model = model
        self.model_as_truth = model_as_truth
        self.const_sv = const_sv

        print("Setting up redshift-space distortions model.")

        # cosmology for Minerva
        self.om_m = om_m
        self.s8 = s8
        self.cosmo = Cosmology(om_m=self.om_m)
        self.nmocks = 299 # hardcoded for Minerva
        self.eff_z = eff_z

        self.growth = self.cosmo.get_growth(self.eff_z)
        self.f = self.cosmo.get_f(self.eff_z)
        self.s8norm = self.s8 * self.growth 

        eofz = np.sqrt((self.om_m * (1 + self.eff_z) ** 3 + 1 - self.om_m))
        self.iaH = (1 + self.eff_z) / (100. * eofz) 

        # read covariance matrix
        if os.path.isfile(covmat_filename):
            print('Reading covariance matrix: ' + covmat_filename)
            self.cov = np.load(covmat_filename)
            self.icov = np.linalg.inv(self.cov)
        else:
            sys.exit('Covariance matrix not found.')

        self.r_for_xi = {}
        self.r_for_delta = {}
        self.s_for_xi = {}
        self.mu_for_xi = {}
        self.delta_r = {}
        self.Delta_r = {}
        self.xi_r = {}
        self.xi0_s = {}
        self.xi2_s = {}

        if self.model == 1 or self.model == 3 or self.model == 4:
            self.r_for_sv = {}
            self.sv = {}
            self.sv_converge = {}

        self.datavec = np.array([])

        for j in range(self.ndenbins):
            denbin = 'den{}'.format(j)
            # read real-space monopole
            data = np.genfromtxt(xi_r_filename[denbin])
            self.r_for_xi[denbin] = data[:,0]
            xi_r = data[:,-2]
            self.xi_r[denbin] = InterpolatedUnivariateSpline(self.r_for_xi[denbin], xi_r, k=3, ext=3)

            # read void-matter correlation function
            data = np.genfromtxt(delta_r_filename[denbin])
            self.r_for_delta[denbin] = data[:,0]
            delta_r = data[:,-2]
            self.delta_r[denbin] = InterpolatedUnivariateSpline(self.r_for_delta[denbin], delta_r, k=3, ext=3)

            # read integrated void-matter correlation function
            data = np.genfromtxt(int_delta_r_filename[denbin])
            self.r_for_delta = data[:,0]
            Delta_r = data[:,-2]
            self.Delta_r[denbin] = InterpolatedUnivariateSpline(self.r_for_delta[denbin], Delta_r, k=3, ext=3)

            if self.model == 1 or self.model == 3 or self.model == 4:
                # read los velocity dispersion profile
                data = np.genfromtxt(sv_filename[denbin])
                self.r_for_sv[denbin] = data[:,0]
                sv = data[:,-2]
                if self.const_sv:
                    sv = np.ones(len(self.r_for_sv[denbin]))
                else:
                    self.sv_converge[denbin] = sv[-1]
                    sv = sv / self.sv_converge
                    sv = savgol_filter(sv, 3, 1)
                self.sv[denbin ]= InterpolatedUnivariateSpline(self.r_for_sv[denbin], sv, k=3, ext=3)


            # read redshift-space correlation function
            self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu_obs = Utilities.readCorrFile(xi_smu_filename[denbin])

            if self.model_as_truth:
                print('Using the model prediction as the measurement.')
                if self.model == 1:
                    fs8 = self.f * self.s8norm
                    sigma_v = self.sv_converge[denbin]
                    alpha = 1.0
                    epsilon = 1.0
                    alpha_para = alpha * epsilon ** (-2/3)
                    alpha_perp = epsilon * alpha_para

                    self.xi0_s[denbin], self.xi2_s[denbin] = self.model1_theory(fs8,
                                                                                sigma_v,
                                                                                alpha_perp,
                                                                                alpha_para,
                                                                                self.s_for_xi[denbin],
                                                                                self.mu_for_xi[denbin],
                                                                                denbin)

            else:
                s, self.xi0_s[denbin] = Utilities.getMonopole(self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu_obs)
                s, self.xi2_s[denbin] = Utilities.getQuadrupole(self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu_obs)


            # restrict measured vectors to the desired fitting scales
            scales = (self.s_for_xi[denbin] >= smin[denbin]) & (self.s_for_xi[denbin] <= smax[denbin])

            self.s_for_xi[denbin] = self.s_for_xi[denbin][scales]
            self.xi0_s[denbin] = self.xi0_s[denbin][scales]
            self.xi2_s[denbin] = self.xi2_s[denbin][scales]

            if self.full_fit:
                self.datavec = np.concatenate((self.datavec, self.xi0_s[denbin], self.xi2_s[denbin]))
            else:
                self.datavec = np.concatenate((self.datavec, self.xi2_s[denbin]))

        

    def log_likelihood(self, theta):
        if self.model == 1:
            if self.ndenbins == 2:
                fs8, sigma_v1, sigma_v2, epsilon = theta
                sigmalist = [sigma_v1, sigma_v2]

            if self.ndenbins == 3:
                fs8, sigma_v1, sigma_v2, sigma_v3, epsilon = theta
                sigmalist = [sigma_v1, sigma_v2, sigma_v3]

        alpha = 1.0
        alpha_para = alpha * epsilon ** (-2/3)
        alpha_perp = epsilon * alpha_para

        sigma_v = {}
        modelvec = np.array([])

        for j in range(self.ndenbins):
            denbin = 'den{}'.format(j)
            sigma_v[denbin] = sigmalist[j]

            xi0, xi2 = self.model1_theory(fs8,
                                          sigma_v[denbin],
                                          alpha_perp,
                                          alpha_para,
                                          self.s_for_xi[denbin],
                                          self.mu_for_xi[denbin],
                                          denbin)

            if self.full_fit:
                modelvec = np.concatenate((modelvec, xi0, xi2))
            else:
                modelvec = np.concatenate((modelvec, xi2))

        chi2 = np.dot(np.dot((modelvec - self.datavec), self.icov), modelvec - self.datavec)
        loglike = -self.nmocks/2 * np.log(1 + chi2/(self.nmocks-1))
        return loglike

    def log_prior(self, theta):
        if self.model == 1:
            if self.ndenbins == 2:
                fs8, sigma_v1, sigma_v2, epsilon = theta

                if 0.1 < fs8 < 2.0 \
                and 1 < sigma_v1 < 500 \
                and 1 < sigma_v2 < 500 \
                and 0.8 < epsilon < 1.2:
                    return 0.0

            if self.ndenbins == 3:
                fs8, sigma_v1, sigma_v2, sigma_v3, epsilon = theta

                if 0.1 < fs8 < 2.0 \
                and 1 < sigma_v1 < 500 \
                and 1 < sigma_v2 < 500 \
                and 1 < sigma_v3 < 500 \
                and 0.8 < epsilon < 1.2:
                    return 0.0

        return -np.inf


    def model1_theory(self, fs8, sigma_v, alpha_perp, alpha_para, s, mu, denbin):
        '''
        RSD model from Nadathur & Percival (2018).
        '''
        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))
        scaled_fs8 = fs8 / self.s8norm

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 80)
        r = self.r_for_delta[denbin]
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r[denbin](r)
        y2 = self.delta_r[denbin](r)
        y3 = self.Delta_r[denbin](r)
        y4 = self.sv[denbin](r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=3)
        rescaled_delta_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=3)
        rescaled_Delta_r = InterpolatedUnivariateSpline(x, y3, k=3, ext=3)
        rescaled_sv = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)
        sigma_v = alpha_para * sigma_v
 

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                rpar = true_spar + true_s * scaled_fs8 * rescaled_Delta_r(true_s) * true_mu[j] / 3.
                sy_central = sigma_v * rescaled_sv(np.sqrt(true_sperp**2 + rpar**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5 * sy_central, 100)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                sy = sigma_v * rescaled_sv(rr) * self.iaH

                integrand = (1 + rescaled_xi_r(rr)) * \
                            (1 + (scaled_fs8 * rescaled_Delta_r(rr) / 3. - y * true_mu[j] / rr) * (1 - true_mu[j]**2) +
                             scaled_fs8 * (rescaled_delta_r(rr) - 2 * rescaled_Delta_r(rr) / 3.) * true_mu[j]**2)
                integrand = integrand * np.exp(-(y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)
                xi_model[j] = np.trapz(integrand, y) - 1


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                  xi_model[np.argsort(true_mu)],
                                                  k=3)
            
            # get multipoles
            xaxis = np.linspace(-1, 1, 1000)

            yaxis = mufunc(xaxis) / 2
            monopole[i] = simps(yaxis, xaxis)

            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / 2
            quadrupole[i] = simps(yaxis, xaxis)
            
            
        return monopole, quadrupole



def SphericalCollapse(g, lna, om_m0, om_l0):
    '''
    Collapse of a spherical shell. Solution to the ODE

    y'' + (1/2 - 3/2 w om_l) y' + om_m/2 (y^{-3} - 1) y = 0

    Let h = y'
    h' + (1/2 - 3/2 w om_l) h + om_m/2 (y^{-3} - 1) y = 0
    '''
    om_m = Omega_m(lna, om_m0=om_m0, om_l0=om_l0)
    om_l = Omega_L(lna, om_m0=om_m0, om_l0=om_l0)
    y, h = g
    dgda = [h, -(1/2 + 3/2*om_l)*h - om_m/2*(y**(-3) - 1)*y]
    return dgda

def Omega_m(lna, om_m0, om_l0):
    a = np.exp(lna)
    om_m = om_m0 / (om_m0 + om_l0 * a**3)
    return om_m

def Omega_L(lna, om_m0, om_l0):
    a = np.exp(lna)
    om_l = om_l0 / (om_m0 * a**-3 + om_l0)
    return om_l

def CosmologicalTime(zi, zf):
    ai = np.log(1/(1 + zi))
    af = np.log(1/(1 + zf))
    t = np.linspace(ai, af, 10000)
    return t

def Hubble(a, om_m0, om_l0):
    return 100 * np.sqrt(om_m0 * a ** -3 + om_l0)




