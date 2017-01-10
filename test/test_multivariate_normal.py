import pickle
import scipy as sp
from pyabc.random_variables import MultivariateMultiTypeNormalDistribution, EmptyMultivariateMultiTypeNormalDistribution, NonEmptyMultivariateMultiTypeNormalDistribution



def test_zero_covariance():
    cov = sp.asarray([[0, 0],
                      [0, 0]])
    par_names = ['a', 'b']
    par_types = [float, float]
    zcvs = 2.3
    m = MultivariateMultiTypeNormalDistribution(cov, par_names, par_types,
                                                zero_covariance_substitutes=zcvs)
    expected = zcvs * sp.eye(2)
    assert sp.absolute(expected-m.covariance_matrix).sum() <  1e-5

def test_pickleable():
    cov = sp.asarray([[0, 0],
                      [0, 0]])

    par_names = ['a', 'b']
    par_types = [float, float]
    zcvs = 2.3
    m = MultivariateMultiTypeNormalDistribution(cov, par_names, par_types,
                                                zero_covariance_substitutes=zcvs)
    s = pickle.dumps(m)
    m_loaded = pickle.loads(s)
    assert m.parameter_names == m_loaded.parameter_names
    assert m.parameter_types == m_loaded.parameter_types
    assert (m.covariance_matrix == m_loaded.covariance_matrix).all()
    assert isinstance(m_loaded, NonEmptyMultivariateMultiTypeNormalDistribution)

def test_empty_multivatiate_pickleable():
    cov = sp.asarray([[]])

    par_names = []
    par_types = []
    zcvs = 2.3
    m = MultivariateMultiTypeNormalDistribution(cov, par_names, par_types,
                                                zero_covariance_substitutes=zcvs)
    s = pickle.dumps(m)
    m_loaded = pickle.loads(s)
    assert isinstance(m_loaded, EmptyMultivariateMultiTypeNormalDistribution)

def test_multiply():
    cov = sp.asarray([[1.1, 0],
                      [0, 2.2]])

    par_names = ['a', 'b']
    par_types = [float, float]
    m = MultivariateMultiTypeNormalDistribution(cov, par_names, par_types)
    m2 = m * 2
    assert 1.1 == m.covariance_matrix[0,0]
    assert 2.2 == m.covariance_matrix[1,1]
    assert 2.2 == m2.covariance_matrix[0,0]
    assert 4.4 == m2.covariance_matrix[1,1]

