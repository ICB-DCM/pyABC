"""Test creation and use of parameters."""

import numpy as np
import pandas as pd
from time import time
import pytest


@pytest.mark.flaky(reruns=2)
def test_parameter_dict_numpy_conversion():
    """Check that the way parameter values are extracted in
    `pyabc.transition.MultivariateNormalTransition.pdf()` is efficient.
    """
    # number of samples
    n = 10000
    # number of parameter keys
    nkey = 500
    # the parameter keys
    keys = ['key_' + str(i) for i in range(int(nkey/2))]
    # the indices
    ixs = list(range(len(keys)))

    np.random.seed(0)

    # create parameters
    pars = []
    start = time()
    for _ in range(n):
        pars.append(dict(zip(keys, np.random.randn(nkey))))
    time_create_dict = time() - start
    print(f"Time to create dict: {time_create_dict}")

    # convert to pandas
    start = time()
    pars_pd = [pd.Series(par) for par in pars]
    time_convert_pd = time() - start
    print(f"Time to convert to pd.Series: {time_convert_pd}")

    # mode 1: extract keys in pandas
    start = time()
    pars_np_1 = [np.array(par[keys]) for par in pars_pd]
    time_np_1 = time() - start
    print(f"Time to extract to numpy via pandas keys: {time_np_1}")

    # mode 2: use cached indices
    start = time()
    pars_np_2 = [np.array(par)[ixs] for par in pars_pd]
    time_np_2 = time() - start
    print(f"Time to extract to numpy via cached indices: {time_np_2}")
    # This is a lot faster than mode 1

    # mode 3: use cached indices and pandas to_numpy
    start = time()
    pars_np_3 = [par.to_numpy()[ixs] for par in pars_pd]
    time_np_3 = time() - start
    print(f"Time to extract to numpy via cached indices and to_numpy(): "
          f"{time_np_3}")
    # This is a little faster than mode 2 (probably just by avoiding copying)

    # mode 4: directly from dict
    start = time()
    pars_np_4 = [np.array([par[key] for key in keys]) for par in pars]
    time_np_4 = time() - start
    print(f"Time to extract directly from dict: {time_np_4}")
    # Taking into account the time to convert to pd, this is faster, however
    # this may change if the values need to be extracted as an array multiple
    # times.

    # check times as expected
    tol = 0.9
    assert time_np_1 > time_np_2 * tol
    assert time_np_2 > time_np_3 * tol
    assert time_np_3 + time_convert_pd > time_np_4 * tol

    # check that we always got the same results
    for par1, par2, par3, par4 in zip(
            pars_np_1, pars_np_2, pars_np_3, pars_np_4):
        assert (par1 == par2).all()
        assert (par1 == par3).all()
        assert (par1 == par4).all()


@pytest.mark.flaky(reruns=2)
def test_sample_selection():
    """Check that the selection of parameters from the population according
    to the weights as in `pyabc.transition.MultivariateNormalTransition.rvs()`
    and `.rvs_single()` is efficient.

    We select a single parameter according to the weights.
    """
    # number of samples
    n = 1000
    # number of parameter keys
    nkey = 50
    # the parameter keys
    keys = ['key_' + str(i) for i in range(int(nkey / 2))]
    # number of repetitions
    nsample = 5000

    np.random.seed(0)

    # create parameters
    pars = [dict(zip(keys, np.random.randn(nkey))) for _ in range(n)]

    # convert population to dataframe
    df = pd.DataFrame(pars)

    # weights
    w = np.random.uniform(size=n)
    w /= w.sum()

    # using pandas
    start = time()
    for _ in range(nsample):
        _ = df.sample(weights=w).iloc[0]
    time_pandas = time() - start
    print(f"Time using pandas sample: {time_pandas}")

    # using numpy
    start = time()
    for _ in range(nsample):
        arr = np.arange(n)
        sample_ind = np.random.choice(arr, p=w, replace=True)
        _ = df.iloc[sample_ind]
    time_numpy = time() - start
    print(f"Time using numpy choice: {time_numpy}")

    # using numpy with caching
    start = time()
    arr = np.arange(n)
    for _ in range(nsample):
        sample_ind = np.random.choice(arr, p=w, replace=True)
        _ = df.iloc[sample_ind]
    time_numpy_cached = time() - start
    print(f"Time using numpy choice cache: {time_numpy_cached}")

    # check times as expected
    tol = 0.9
    assert time_numpy < time_pandas
    assert time_numpy_cached * tol < time_numpy
