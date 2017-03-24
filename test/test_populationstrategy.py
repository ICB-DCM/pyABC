import pytest
from pyabc.populationstrategy import (AdaptivePopulationStrategy,
                                      ConstantPopulationStrategy,
                                      PopulationStrategy)
from pyabc.transition import MultivariateNormalTransition
import pandas as pd
import scipy as sp


def Adaptive():
    return AdaptivePopulationStrategy(100, 2)


def Constant():
    return ConstantPopulationStrategy(100,  2)


@pytest.fixture(params=[Adaptive, Constant])
def population_strategy(request):
    return request.param()


def test_adapt_single_model(population_strategy: PopulationStrategy):
    n = 10
    df = pd.DataFrame([{"s": sp.rand()} for _ in range(n)])
    w = sp.ones(n) / n
    kernel = MultivariateNormalTransition()
    kernel.fit(df, w)

    population_strategy.adapt_population_size([kernel], sp.array([1.]))
    assert population_strategy.nr_particles > 0


def test_adapt_two_models(population_strategy: PopulationStrategy):
    n = 10
    kernels = []
    for _ in range(2):
        df = pd.DataFrame([{"s": sp.rand()} for _ in range(n)])
        w = sp.ones(n) / n
        kernel = MultivariateNormalTransition()
        kernel.fit(df, w)
        kernels.append(kernel)

    population_strategy.adapt_population_size(kernels, sp.array([.7, .2]))
    assert population_strategy.nr_particles > 0


def test_no_parameters(population_strategy: PopulationStrategy):
    n = 10
    df = pd.DataFrame(index=list(range(n)))
    w = sp.ones(n) / n

    kernels = []
    for _ in range(2):
        kernel = MultivariateNormalTransition()
        kernel.fit(df, w)
        kernels.append(kernel)

    population_strategy.adapt_population_size(kernels, sp.array([.7, .3]))
    assert population_strategy.nr_particles > 0
