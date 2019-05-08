import numpy as np

def smfilt(df):
    y = df['y']
    ll = len(y)
    TT = df['TT']
    lTT = df.shape[1]
    ncA0 = df['ncA0']
    N = np.zeros((lTT, ncA0))
    R = np.zeros(0, lTT, lTT)
    Z = dk['Z']
    ncolZ = Z.shape[1]
    gamma_est = dkf.out['gamma.est']

    
    mse.gamma = as.matrix(dkf.out$mse.gamma)
    sigma2 = as.numeric(dkf.out$sigma2)
    alphahat = matrix(0, ncolZ, ll)
    mse.alphahat = matrix(0, lTT, lTT * ll)
    for (i in 1:ll){
        tZ = t(Z)
    D = dkf.out$D[(ll-i+1)]
    E = dkf.out$E[, ((ll * ncA0-ncA0 * i+1):(ncA0 * ll - ncA0 * (i - 1)))]
    A = dkf.out$A[, ((ll * ncA0 - ncA0 * i + 1):(ncA0 * ll-ncA0 * (i-1)))]
    P = dkf.out$P[, ((ll * ncolZ - ncolZ * i + 1):(ncolZ * ll-ncolZ * (i-1)))]
    K = matrix(dkf.out$K[, (ll - i + 1)], ncolZ, 1)
    L = TT - K % * % Z
    Dinv = 1 / D
    Dinv = as.numeric(Dinv)
    tL = t(L)
    junk = Dinv * tZ
    N = junk % * % E + tL % * % N
    R = Dinv * (tZ % * % Z) + tL % * % R % * % L
    Naux = (A + P % * % N)
    Naux2 = Naux % * % matrix(c(-gamma.est, 1), ncA0, 1)
    alphahat[, ll - i + 1] = Naux2
    Naux2 = matrix(Naux[, (1:(ncA0-1))], lTT, (ncA0 - 1))
    mse = (sigma2 * (P - P % * % R % * % P)) + (Naux2 % * % mse.gamma % * % t(Naux2))
    mse.alphahat[, (lTT * (ll - i) + 1): (lTT * (ll - i + 1))] = mse
    }
    list(alpha=alphahat, mse.alpha = mse.alphahat)
    }
