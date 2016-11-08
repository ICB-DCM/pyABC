from pyabc.transition import variance, MultivariateNormalTransition
import pandas as pd
import numpy as np


def data(n):
    df = pd.DataFrame({"a": np.random.rand(n), "b": np.random.rand(n)})
    w = np.ones(len(df)) / len(df)
    return df, w


def test_variance_estimate():
    np.random.seed(42)
    var_list = []
    for n in [10, 50, 100]:
        m = MultivariateNormalTransition()
        df, w = data(n)
        var = variance(m, df, w)
        var_list.append(var)

    print(var_list)
    for higher, lower in zip(var_list[:-1], var_list[1:]):
        assert higher >= lower


def test_variance_no_side_effect():
    m = MultivariateNormalTransition()
    df, w = data(10)
    m.fit(df, w)
    # very intrusive test. touches intermals of m. not very nice.
    X_orig_id = id(m.X)
    var = variance(m, df, w)
    assert id(m.X) == X_orig_id


def test_argument_order():
    """
    Dataframes passed to the transition kernels are generated from dicts.
    Order of parameter names is no guaranteed.
    The Transition kernel has to take care of the correct sorting.
    """
    m = MultivariateNormalTransition()
    df, w = data(20)
    m.fit(df, w)
    test = df.iloc[0]
    reversed = test[::-1]
    assert (np.array(test) != np.array(reversed)).all()   # works b/c of even nr of parameters
    assert m.pdf(test) == m.pdf(reversed)
