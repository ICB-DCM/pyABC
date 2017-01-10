from collections import OrderedDict
import scipy as sp
from pyabc.legacy_covariance import cov as cov_func


def test_covariance_single_variable_two_samples():
    particles = [{"weight": .5, "parameter": OrderedDict({"par1": -1})},
                 {"weight": .5, "parameter": OrderedDict({"par1": 1})}]
    expected = sp.cov([-1,1], aweights=[1/2, 1/2], ddof=0)
    cov = cov_func(particles)
    cov = cov.covariance_matrix
    assert expected == cov

def test_covariance_single_variable_three_samples():
    particles = [{"weight": 1/3, "parameter": OrderedDict({"par1": -1})},
                 {"weight": 1/3, "parameter": OrderedDict({"par1": 0})},
                 {"weight": 1/3, "parameter": OrderedDict({"par1": 1})}]
    expected = sp.cov([-1,0,1], aweights=[1/3, 1/3, 1/3], ddof=0)
    cov = cov_func(particles)
    cov = cov.covariance_matrix
    assert expected == cov

def test_covariance_single_variable_not_symmetric():
    particles = [{"weight": .5, "parameter": OrderedDict({"par1": 0})},
                 {"weight": .5, "parameter": OrderedDict({"par1": 1})}]
    expected = sp.cov([0, 1], aweights=[1/2, 1/2], ddof=0)
    cov = cov_func(particles)
    cov = cov.covariance_matrix
    assert expected == cov

def test_covariance_two_variables():
    particles = [{"weight": .5, "parameter": OrderedDict(par1=-1, par2=2)},
                 {"weight": .5, "parameter": OrderedDict(par1=1, par2=3)}]
    expected = sp.cov([[-1, 1],
                       [2, 3]], aweights=[.5,.5], ddof=0)
    cov = cov_func(particles)
    cov = cov.covariance_matrix
    assert sp.isclose(expected, cov, rtol=1e-3).all()

def test_covariance_two_variables_weighted():
    particles = [{"weight": .3, "parameter": OrderedDict(par1=-1, par2=2)},
                 {"weight": .7, "parameter": OrderedDict(par1=1, par2=3)}]
    expected = sp.cov([[-1, 1],
                       [2, 3]], aweights=[.3, .7], ddof=0)
    cov = cov_func(particles)

    cov = cov.covariance_matrix
    assert sp.isclose(expected, cov, rtol=1e-3).all()

def test_covariance_two_variables_weighted_three_samples():
    particles = [{"weight": .3, "parameter": OrderedDict(par1=-1, par2=2)},
                 {"weight": .2, "parameter": OrderedDict(par1=-1, par2=2)},
                 {"weight": .5, "parameter": OrderedDict(par1=10, par2=-5)}]
    expected = sp.cov([[-1, -1, 10],
                       [2, 2, -5]], aweights=[.3, .2, .5], ddof=0)
    cov = cov_func(particles)

    cov = cov.covariance_matrix
    assert sp.isclose(expected, cov, rtol=1e-3).all()

