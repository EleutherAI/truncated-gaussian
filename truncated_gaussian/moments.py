import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve

from .marginals import dtmvnorm_marginal, dtmvnorm_marginal2


def JohnsonKotzFormula(mean, sigma, lower, upper):
    idx = np.where(np.logical_or(np.isfinite(lower), np.isfinite(upper)))[0]
    n = len(mean)
    k = len(idx)

    if k >= n:
        raise ValueError("Can't truncate more than the total number of variables.")

    if k == 0:
        return {"tmean": mean, "tvar": sigma}

    # Transform to zero mean first
    lower = lower - mean
    upper = upper - mean

    V11 = sigma[np.ix_(idx, idx)]
    V12 = sigma[np.ix_(idx, np.setdiff1d(np.arange(n), idx))]
    V21 = sigma[np.ix_(np.setdiff1d(np.arange(n), idx), idx)]
    V22 = sigma[
        np.ix_(np.setdiff1d(np.arange(n), idx), np.setdiff1d(np.arange(n), idx))
    ]

    r = mtmvnorm(mean=np.zeros(k), sigma=V11, lower=lower[idx], upper=upper[idx])
    xi = r["tmean"]
    U11 = r["tvar"]

    invV11 = solve(V11, np.eye(V11.shape[0]))

    tmean = np.zeros(n)
    tmean[idx] = xi
    tmean[np.setdiff1d(np.arange(n), idx)] = xi @ invV11 @ V12

    tvar = np.zeros((n, n))
    tvar[np.ix_(idx, idx)] = U11
    tvar[np.ix_(idx, np.setdiff1d(np.arange(n), idx))] = U11 @ invV11 @ V12
    tvar[np.ix_(np.setdiff1d(np.arange(n), idx), idx)] = V21 @ invV11 @ U11
    tvar[np.ix_(np.setdiff1d(np.arange(n), idx), np.setdiff1d(np.arange(n), idx))] = (
        V22 - V21 @ (invV11 - invV11 @ U11 @ invV11) @ V12
    )

    tmean = tmean + mean

    return {"tmean": tmean, "tvar": tvar}


def mtmvnorm_quadrature(mean, sigma, lower, upper):
    k = len(mean)

    def expectation(x, n=1):
        return x * dtmvnorm_marginal(x, n, mean, sigma, lower, upper)

    def variance(x, n=1):
        return (x - m_integration[n]) ** 2 * dtmvnorm_marginal(
            x, n, mean, sigma, lower, upper
        )

    m_integration = np.zeros(k)
    for i in range(k):
        m_integration[i], _ = quad(expectation, lower[i], upper[i], args=(i + 1,))

    v_integration = np.zeros(k)
    for i in range(k):
        v_integration[i], _ = quad(variance, lower[i], upper[i], args=(i + 1,))

    return {"m": m_integration, "v": v_integration}


# Helper function to check the input arguments for tmv related functions
def check_tmv_args(mean, sigma, lower, upper):
    mean = np.array(mean)
    sigma = np.array(sigma)
    lower = np.array(lower)
    upper = np.array(upper)

    # Some basic validation
    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Sigma must be a square matrix")

    if mean.shape[0] != sigma.shape[0]:
        raise ValueError(
            "Length of the mean vector must match the dimensions of the sigma matrix"
        )

    if len(lower) != len(upper):
        raise ValueError("Lower and upper bounds must have the same length")

    if len(lower) != len(mean):
        raise ValueError(
            "Lower and upper bounds must have the same length as the mean vector"
        )

    return {"mean": mean, "sigma": sigma, "lower": lower, "upper": upper}


def mtmvnorm(mean, sigma, lower, upper, do_compute_variance=True):
    # Check input parameters
    check_tmv_args(mean, sigma, lower, upper)
    N = len(mean)

    # Convert lists to NumPy arrays if they are not already
    mean = np.array(mean)
    sigma = np.array(sigma)
    lower = np.array(lower)
    upper = np.array(upper)

    # Initialize truncated mean and variance
    tmean = np.zeros(N)
    tvar = np.zeros((N, N))

    # Shift the integration bounds by -mean to make the mean zero
    a = lower - mean
    b = upper - mean
    lower = a
    upper = b

    # Pre-calculate one-dimensional marginals F_a and F_b
    F_a = np.zeros(N)
    F_b = np.zeros(N)

    zero_mean = np.zeros(N)

    # Pre-calculate one-dimensional marginals F_a once
    for q in range(N):
        tmp = dtmvnorm_marginal(
            np.array([a[q], b[q]]),
            n=q,
            mean=zero_mean,
            sigma=sigma,
            lower=lower,
            upper=upper,
        )
        F_a[q] = tmp[0]
        F_b[q] = tmp[1]

    # Compute truncated mean
    tmean = sigma @ (F_a - F_b)

    if do_compute_variance:
        # Compute truncated variance
        # TODO: The original R code includes various optimizations and special cases.
        # Here we'll just illustrate core logic for computing the truncated variance.

        F2 = np.zeros((N, N))
        for q in range(N):
            for s in range(N):
                if q != s:
                    d = dtmvnorm_marginal2(
                        np.array([a[q], b[q], a[q], b[q]]),
                        np.array([a[s], a[s], b[s], b[s]]),
                        q,
                        s,
                        zero_mean,
                        sigma,
                        lower,
                        upper,
                    )
                    F2[q, s] = (d[0] - d[1]) - (d[2] - d[3])

        # Compute the truncated variance
        for i in range(N):
            for j in range(N):
                summation = 0

                for q in range(N):
                    summation += (sigma[i, q] * sigma[j, q] / sigma[q, q]) * (
                        a[q] * F_a[q] - b[q] * F_b[q]
                    )

                    if j != q:
                        sum2 = 0
                        for s in range(N):
                            tt = sigma[j, s] - sigma[q, s] * sigma[j, q] / sigma[q, q]
                            sum2 += tt * F2[q, s]

                        sum2 = sigma[i, q] * sum2
                        summation += sum2

                tvar[i, j] = sigma[i, j] + summation

        # Finalize the truncated variance
        tvar = tvar - np.outer(tmean, tmean)

    # Shift back by the original mean
    tmean = tmean + mean

    return {"tmean": tmean, "tvar": tvar}
