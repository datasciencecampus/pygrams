import numpy as np
import pandas as pd
from math import sqrt
from scipy.optimize import minimize
from itertools import product


class StateSpaceModel:
    def __init__(self, timeseries):
        self.timeseries = timeseries

    @staticmethod
    def expand_grid(dictionary):

        return pd.DataFrame([row for row in product(*dictionary.values())],
                            columns=dictionary.keys())

    def forecast(self, dkf_in, k):

        A = dkf_in['Alast'].values[0]
        P = dkf_in['Plast'].values[0]
        TT = dkf_in['TT'].values[0]
        Z = dkf_in['Z'].values[0]
        ncA0 = dkf_in['ncA0'].values[0]
        gamma = dkf_in['gamma_est'].values[0]
        mse_gamma = dkf_in['mse_gamma'].values[0]
        sigma2 = dkf_in['sigma2'].values[0]
        ncolZ = Z.shape[1]

        Agamma = A[:, :ncA0 - 1]
        alpha_hat     = np.matrix(np.vstack([-999.0]*ncolZ * k).reshape(ncolZ, k))
        mse_alpha_hat = np.matrix(np.vstack([-999.0]*ncolZ*ncolZ * k).reshape(ncolZ, ncolZ * k))

        aux = np.vstack([-gamma, 1]).reshape(ncA0, 1)
        alpha = np.matmul(A,  aux)
        alpha_hat[: , 0] = alpha

        m1 = np.matmul(Agamma, mse_gamma)
        m2 = np.matmul(m1, np.transpose(Agamma))
        msealpha = (sigma2 * P) + m2

        mse_alpha_hat[:, 0:ncolZ] = msealpha
        if k > 1:
            for i in range(1,k):
                alpha_hat[:, i] = TT * alpha
                mse_alpha_hat[: , ((i - 1) * ncolZ ): (ncolZ * i)] = TT * msealpha * TT.transpose()
                TT = np.matmul(TT,  TT)
            return alpha_hat, mse_alpha_hat
        return None, None

    # Calculating and smoothing quarterly timeseries:   1%|          | 1.08k/100k [23:31<36:43:32, 1.34s/term]
    def param_estimator(self, sigma_gnu, sigma_eta, delta):
        param_dictionary = {'sigma_gnu': sigma_gnu, 'sigma_eta': sigma_eta, 'delta': delta}
        param_grid = self.expand_grid(param_dictionary)
        res = []

        for index, row in param_grid.iterrows():
            params = [row['sigma_gnu'], row['sigma_eta'], row['delta']]
            res.append(self.lik_llm_vard(params))

        idx = res.index(min(res))
        optimal_param = minimize(self.lik_llm_vard, param_grid.loc[idx, :].tolist(), method='L-BFGS-B')['x']

        return optimal_param

    def lik_llm_vard(self, params, sigma_eps=1, rho=1):
        sigma_eps = sigma_eps
        sigma_gnu = params[0]
        sigma_eta = params[1]
        delta = params[2]
        rho = rho
        timeseries_length = len(self.timeseries)
        GGt = sigma_eps ** 2
        aux = [rho, 1, 0, delta]
        TT = np.matrix(aux).reshape((2, 2))
        lTT = TT.shape[0]
        A = np.c_[-np.diag(np.ones(lTT)), np.zeros(lTT)]
        ncA0 = A.shape[1]
        QQ = np.matrix(np.zeros(ncA0 * ncA0).reshape(ncA0, ncA0))
        P0 = np.matrix(np.zeros(lTT * lTT)).reshape(lTT, lTT)
        P = P0
        Z = np.matrix([1, 0]).reshape(1, lTT)
        Z_trans = np.transpose(Z)
        DD = np.zeros(timeseries_length)
        elem = [(sigma_gnu ** 2), 0, 0, (sigma_eta ** 2)]
        HHt = np.matrix(elem).reshape(lTT, lTT)
        np_matrix_zeros_ncA0 = np.matrix(np.zeros(ncA0 - 1))

        for i in range(0, timeseries_length):
            aux2 = np.c_[np_matrix_zeros_ncA0, self.timeseries[i]].reshape(1, ncA0)
            E = aux2 - np.matmul(Z, A)
            D = np.matmul(Z, np.matmul(P, Z_trans)) + GGt
            DD[i] = D
            Dinv = 1 / D
            K = np.matmul(TT, np.matmul(P, np.matmul(Z_trans, Dinv)))
            A = np.matmul(TT, A) + np.matmul(K, E)
            L = TT - np.matmul(K, Z)
            P = np.matmul(L, np.matmul(P, np.transpose(TT))) + HHt
            QQ = QQ + np.matmul(np.transpose(E), np.matmul(Dinv, E))

        SS = QQ[0:lTT, 0:lTT]
        qqq = np.matrix(QQ[lTT:ncA0, lTT:ncA0])
        ss = np.asarray(QQ[0:lTT, lTT:ncA0])
        Sinv = np.linalg.pinv(SS)
        sigma2_est = (qqq - (np.matmul(np.transpose(ss), np.matmul(Sinv, ss)))) / timeseries_length
        sigma2_est = sigma2_est[0, 0]
        sigmatilde2 = (timeseries_length / (timeseries_length - ncA0 + 1)) * sigma2_est
        loglik = (-0.5) * ((timeseries_length - ncA0 + 1) * (1 + np.log(sigmatilde2)) + sum(np.log(abs(DD))))

        return -loglik

    def dfk_llm_vard(self, params, sigma_eps=1, rho=1):
        sigma_eps = sigma_eps
        sigma_gnu = params[0]
        sigma_eta = params[1]
        delta = params[2]
        rho = rho
        ll = len(self.timeseries)
        GGt = sigma_eps ** 2
        aux = [rho, 1, 0, delta]
        TT = np.matrix(aux).reshape((2, 2))
        lTT = TT.shape[0]
        A = np.c_[-np.diag(np.ones(lTT)), np.zeros(lTT)]
        ncA0 = A.shape[1]
        QQ = np.matrix(np.zeros(ncA0 * ncA0).reshape(ncA0, ncA0))
        Z = np.matrix([1, 0]).reshape(1, lTT)
        DD = np.zeros(ll)
        EE = np.matrix(np.zeros(ncA0 * ll))
        AA = np.matrix(np.zeros(ncA0 * lTT * (ll + 1))).reshape(lTT, ncA0 * (ll + 1))
        AA[:, 0:ncA0] = A
        PP = np.matrix(np.zeros(lTT * lTT * (ll + 1))).reshape(lTT, lTT * (ll + 1))
        KK = np.matrix(np.zeros(lTT * ll)).reshape(lTT, ll)
        elem = [sigma_gnu ** 2, 0, 0, sigma_eta ** 2]
        HHt = np.matrix(elem).reshape(lTT, lTT)
        P = np.zeros(lTT * lTT).reshape(lTT, lTT)
        PP[:, 0:lTT] = P
        ee = np.zeros(ll)
        mse_ee = np.zeros(ll)

        for i in range(0, ll):
            Z = np.matrix([1, 0]).reshape(1, lTT)
            aux2 = np.c_[np.matrix(np.zeros(ncA0 - 1)), self.timeseries[i]].reshape(1, ncA0)
            E = aux2 - np.matmul(Z, A)
            EE[:, (i * ncA0):((i + 1) * ncA0)] = E
            EEgam = np.matrix(E[:, 0:lTT]).reshape(1, lTT)
            D = np.matmul(Z, np.matmul(P, np.transpose(Z))) + GGt
            DD[i] = D
            Dinv = 1 / D
            K = np.matmul(TT, np.matmul(P, np.matmul(np.transpose(Z), Dinv)))
            KK[:, i] = K
            A = np.matmul(TT, A) + np.matmul(K, E)
            AA[:, (i + 1) * ncA0:(i + 2) * ncA0] = A
            L = TT - np.matmul(K, Z)
            P = np.matmul(L, np.matmul(P, np.transpose(TT))) + HHt
            PP[:, ((i + 1) * lTT):((i + 1) * lTT + 2)] = P

            if i > 1:
                SS = QQ[0:lTT, 0:lTT]
                ss = np.asarray(QQ[0:lTT, lTT:ncA0])
                SSinv = np.linalg.pinv(SS)
                gamma = np.matmul(SSinv, ss)
                aux = np.vstack([-gamma, 1]).reshape(ncA0, 1)
                ee[i] = np.matmul(E, aux)
                mse_ee[i] = np.matmul(EEgam, np.matmul(SSinv, np.transpose(EEgam)))

            QQ = QQ + np.matmul(np.transpose(E), np.matmul(Dinv, E))

        Alast = A
        Plast = P
        SS = QQ[0:lTT, 0:lTT]
        aux = np.linalg.eig(SS)[0]
        ldSS = sum(np.log(aux))
        qqq = np.matrix(QQ[lTT:ncA0, lTT:ncA0])
        ss = np.asarray(QQ[0:lTT, lTT:ncA0])
        Sinv = np.linalg.pinv(SS)
        gamma_est = np.matmul(Sinv, ss)
        sigma2_est = (qqq - (np.matmul(np.transpose(ss), np.matmul(Sinv, ss)))) / ll
        sigma2_est = sigma2_est[0, 0]
        sigmatilde2 = (ll / (ll - ncA0 + 1)) * sigma2_est
        mse_gamma_est = sigma2_est * Sinv
        mse_ee = sigma2_est * (DD + mse_ee)
        l0 = (-0.5) * (ll * (1 + np.log(sigma2_est)) + sum(np.log(abs(DD))))
        linfty = (-0.5) * ((ll - ncA0 + 1) * (1 + np.log(sigmatilde2)) + ldSS + sum(np.log(abs(DD))))
        ee_std = ee / np.sqrt(mse_ee)

        dfk_out = pd.DataFrame({'gamma_est': [gamma_est],
                                'mse_gamma': [mse_gamma_est],
                                'linfty': [linfty],
                                'l0': [l0],
                                'ncA0': [ncA0],
                                'll': [ll],
                                'delta': [delta],
                                'D': [DD],
                                'E': [EE],
                                'Z': [Z],
                                'A': [AA],
                                'P': [PP],
                                'K': [KK],
                                'TT': [TT],
                                'y': [self.timeseries],
                                'ee': [ee],
                                'mse_ee': [mse_ee],
                                'ee_std': [ee_std],
                                'Alast': [Alast],
                                'Plast': [Plast],
                                'sigma2': [sigma2_est],
                                'sigma_eps2': [sigma2_est * sigma_eps ** 2],
                                'sigma_gnu2': [sigma2_est * sigma_gnu ** 2],
                                'sigma_eta2': [sigma2_est * sigma_eta ** 2],
                                'sigma_eps': [np.sqrt(sigma2_est * sigma_eps ** 2)],
                                'sigma_gnu': [np.sqrt(sigma2_est * sigma_gnu ** 2)],
                                'sigma_eta': [np.sqrt(sigma2_est * sigma_eta ** 2)]})

        return dfk_out

    def smfilt(self, dkf_out):
        y = self.timeseries
        ll = len(y)
        TT = dkf_out['TT'].values[0]
        lTT = TT.shape[1]
        ncA0 = dkf_out['ncA0'].values[0]
        N = np.zeros(lTT * ncA0).reshape(lTT, ncA0)
        R = np.zeros(lTT * lTT).reshape(lTT, lTT)
        Z = dkf_out['Z'].values[0]
        ncolZ = Z.shape[1]
        gamma_est = dkf_out['gamma_est'].values[0]
        mse_gamma = dkf_out['mse_gamma'].values[0]
        sigma2 = dkf_out['sigma2'].values[0]
        alphahat = np.matrix(np.zeros(ncolZ * ll).reshape(ncolZ, ll))
        mse_alphahat = np.zeros(lTT * (lTT * ll)).reshape(lTT, (lTT * ll))
        tZ = np.transpose(Z)

        for i in range(0, ll):
            D = dkf_out['D'].values[0][ll - (i + 1)]
            E = dkf_out['E'].values[0][:, ll * ncA0 - ncA0 * (i + 1):ncA0 * ll - ncA0 * i]
            A = dkf_out['A'].values[0][:, ll * ncA0 - ncA0 * (i + 1):ncA0 * ll - ncA0 * i]
            P = dkf_out['P'].values[0][:, ll * ncolZ - ncolZ * (i + 1):ncolZ * ll - ncolZ * i]
            K = dkf_out['K'].values[0][:, (ll - i - 1)].reshape(ncolZ, 1)
            L = TT - np.matmul(K, Z)
            Dinv = 1 / D
            tL = np.transpose(L)
            junk = Dinv * tZ
            N = np.matmul(junk, E) + np.matmul(tL, N)
            R = Dinv * (np.matmul(tZ, Z)) + np.matmul(tL, np.matmul(R, L))
            naux = A + np.matmul(P, N)
            naux2 = np.matmul(naux, np.vstack([-gamma_est, 1]).reshape(ncA0, 1))
            alphahat[:, ll - i - 1] = naux2
            naux2 = naux[:, 0:(ncA0 - 1)].reshape(lTT, (ncA0 - 1))
            mse = (sigma2 * (P - np.matmul(P, np.matmul(R, P))) + (
                np.matmul(naux2, np.matmul(mse_gamma, np.transpose(naux2)))))
            mse_alphahat[:, (lTT * (ll - i - 1)): (lTT * (ll - i))] = mse

        return alphahat, mse_alphahat

    def run_smoothing(self, sigma_gnu=[0.001, 0.01, 0.1], sigma_eta=[0.001, 0.01, 0.1], delta=[0.5, 0.9], forecast = False):
        opt_param = self.param_estimator(sigma_gnu, sigma_eta, delta)
        dfk_out = self.dfk_llm_vard(opt_param)
        if forecast:
            yhat, mse_yhat = self.forecast(dfk_out, 5)
            return  yhat, mse_yhat, None, None
        alphahat, mse_alphahat = self.smfilt(dfk_out)

        MSE = sum([((x-y)*(x-y)) for x,y in zip(self.timeseries, alphahat[0].tolist()[0])])/len(self.timeseries)

        if MSE < 1.0 and sum(self.timeseries)/len(self.timeseries) > 1.5:
            sigma_gnu = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5,1.0, 1.5]
            sigma_eta = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5,1.0, 1.5]
            delta = [0.3, 0.6, 0.9, 1.2, 1.5]
            opt_param = self.param_estimator(sigma_gnu, sigma_eta, delta)
            dfk_out = self.dfk_llm_vard(opt_param)
            alphahat, mse_alphahat = self.smfilt(dfk_out)
        return opt_param, dfk_out, alphahat, mse_alphahat
