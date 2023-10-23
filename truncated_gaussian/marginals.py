from math import exp, pi, sqrt

import numpy as np
from scipy.linalg import eigvalsh, solve
from scipy.stats import mvn, norm


def dtmvnorm_marginal(xn, mean, sigma, lower, upper, n: int = 0, log_bool=False):
    k = len(mean)

    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square matrix")

    if len(mean) != sigma.shape[0]:
        raise ValueError("mean and sigma have non-conforming size")

    if n < 0 or n >= len(mean):
        raise ValueError("n must be an integer scalar in 0..len(mean)-1")

    if k == 1:
        prob = norm.cdf(upper, loc=mean, scale=sqrt(sigma)) - norm.cdf(
            lower, loc=mean, scale=sqrt(sigma)
        )
        density = np.where(
            (lower[0] <= xn) & (xn <= upper[0]),
            norm.pdf(xn, loc=mean, scale=sqrt(sigma)) / prob,
            0,
        )

        return np.log(density) if log_bool else density

    C = sigma
    A = solve(C, np.identity(k))

    A_1 = np.delete(np.delete(A, n, 0), n, 1)
    A_1_inv = solve(A_1, np.identity(k - 1))

    np.delete(np.delete(C, n, 0), n, 1)
    c_nn = C[n, n]
    c = C[:, n]

    mu = mean
    mu_1 = np.delete(mu, n)
    mu_n = mu[n]

    p, _ = mvn.mvnun(lower, upper, mu, C)

    f_xn = []
    for xi in xn:
        if xi < lower[n] or xi > upper[n] or np.isinf(xi):
            f_xn.append(0)
            continue

        m = mu_1 + (xi - mu_n) * c[:-1] / c_nn
        mvn_cdf, _ = mvn.mvnun(lower[:-1], upper[:-1], m, A_1_inv)
        f_xn.append(exp(-0.5 * (xi - mu_n) ** 2 / c_nn) * mvn_cdf)

    density = 1 / p * 1 / sqrt(2 * pi * c_nn) * np.array(f_xn)
    return np.log(density) if log_bool else density


def dmvnorm(x, mean, sigma, log_bool=False):
    distval = np.sum(
        (x - mean).dot(solve(sigma, np.eye(len(mean)))) * (x - mean), axis=1
    )
    logdet = np.sum(np.log(eigvalsh(sigma)))
    logretval = -(x.shape[1] * np.log(2 * np.pi) + logdet + distval) / 2
    return logretval if log_bool else np.exp(logretval)


# Complete the translation of R function dtmvnorm.marginal2 to Python
def dtmvnorm_marginal2(xq, xr, q, r, mean, sigma, lower, upper, log_bool=False):
    n = sigma.shape[0]
    N = len(xq)

    if n < 2:
        raise ValueError("Dimension n must be >= 2!")

    alpha, _ = mvn.mvnun(lower, upper, mean, sigma)

    if n == 2:
        out_of_bounds = (
            (xq < lower[q]) | (xq > upper[q]) | (xr < lower[r]) | (xr > upper[r])
        )
        density = np.zeros(N)
        density[out_of_bounds] = 0
        density[~out_of_bounds] = (
            dmvnorm(
                np.column_stack((xq[~out_of_bounds], xr[~out_of_bounds])),
                mean[[q, r]],
                sigma[np.ix_([q, r], [q, r])],
            )
            / alpha
        )
        return np.log(density) if log_bool else density

    # Standard deviation for normalization
    SD = np.sqrt(np.diag(sigma))

    # Normalized bounds
    lower_normalised = (lower - mean) / SD
    upper_normalised = (upper - mean) / SD

    xq_normalised = (xq - mean[q]) / SD[q]
    xr_normalised = (xr - mean[r]) / SD[r]

    # Computing correlation matrix R from sigma
    D = np.diag(1 / SD)
    R = D.dot(sigma).dot(D)

    # Determine (n-2) x (n-2) correlation matrix RQR
    RQR = np.zeros((n - 2, n - 2))
    R_inv = solve(R, np.eye(n))
    WW = np.zeros((n - 2, n - 2))

    M1 = 0
    for i in range(n):
        if i != q and i != r:
            M1 += 1
            M2 = 0
            for j in range(n):
                if j != q and j != r:
                    M2 += 1
                    WW[M1 - 1, M2 - 1] = R_inv[i, j]

    WW = solve(WW, np.eye(n - 2))
    for i in range(n - 2):
        for j in range(n - 2):
            RQR[i, j] = WW[i, j] / np.sqrt(WW[i, i] * WW[j, j])

    # Determine bounds of integration vector AQR and BQR (n - 2) x 1
    AQR = np.zeros((N, n - 2))
    BQR = np.zeros((N, n - 2))

    M2 = 0
    for i in range(n):
        if i != q and i != r:
            M2 += 1
            BSQR = (R[q, i] - R[q, r] * R[r, i]) / (1 - R[q, r] ** 2)
            BSRQ = (R[r, i] - R[q, r] * R[q, i]) / (1 - R[q, r] ** 2)
            RSRQ = (1 - R[i, q] ** 2) * (1 - R[q, r] ** 2)
            RSRQ = (R[i, r] - R[i, q] * R[q, r]) / np.sqrt(RSRQ)

            # Lower integration bound
            AQR[:, M2 - 1] = (
                lower_normalised[i] - BSQR * xq_normalised - BSRQ * xr_normalised
            ) / np.sqrt((1 - R[i, q] ** 2) * (1 - RSRQ**2))

            # Upper integration bound
            BQR[:, M2 - 1] = (
                upper_normalised[i] - BSQR * xq_normalised - BSRQ * xr_normalised
            ) / np.sqrt((1 - R[i, q] ** 2) * (1 - RSRQ**2))

            AQR[:, M2 - 1] = np.where(np.isnan(AQR[:, M2 - 1]), -np.inf, AQR[:, M2 - 1])
            BQR[:, M2 - 1] = np.where(np.isnan(BQR[:, M2 - 1]), np.inf, BQR[:, M2 - 1])

    density = np.zeros(N)

    for i in range(N):
        if xq[i] < lower[q] or xq[i] > upper[q] or xr[i] < lower[r] or xr[i] > upper[r]:
            density[i] = 0
        else:
            if (n - 2) == 1:
                prob, _ = mvn.mvnun(AQR[i, :], BQR[i, :], np.zeros(n - 2), RQR)
            else:
                prob, _ = mvn.mvnun(AQR[i, :], BQR[i, :], np.zeros(n - 2), RQR)

            density[i] = (
                dmvnorm(
                    np.array([[xq[i], xr[i]]]),
                    mean[[q, r]],
                    sigma[np.ix_([q, r], [q, r])],
                )
                * prob
                / alpha
            )

    return np.log(density)
