import numpy as np

from truncated_gaussian.moments import mtmvnorm


def test_mtmvnorm():
    mean_vector = [0, 0]
    sigma_matrix = [[1, 0.5], [0.5, 1]]
    lower_bounds = [-1, -1]
    upper_bounds = [1, 1]

    result = mtmvnorm(mean_vector, sigma_matrix, lower_bounds, upper_bounds)
    np.testing.assert_allclose(result["tmean"], [0, 0], atol=1e-5)
    np.testing.assert_allclose(
        result["tvar"], [[0.28282885, 0.05184094], [0.05184094, 0.28282885]], atol=1e-5
    )
