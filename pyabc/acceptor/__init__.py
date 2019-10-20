"""
Acceptor
========

<<<<<<< HEAD
Acceptors handle the acceptance step. Stochastic acceptors make use of
temperature schemes and pdf_max_eval methods.
=======
Acceptors handle the acceptance step.
>>>>>>> develop

"""


from .acceptor import (
    AcceptorResult,
    Acceptor,
    SimpleFunctionAcceptor,
    UniformAcceptor,
    StochasticAcceptor,)
from .temperature_scheme import (
    scheme_acceptance_rate,
    scheme_polynomial_decay,
    scheme_exponential_decay,
    scheme_daly,
    scheme_ess,
    scheme_friel_pettitt,)
from .pdf_max_eval import (
    pdf_max_take_from_kernel,
    pdf_max_take_max_found,)


__all__ = [
    # acceptor
    'AcceptorResult',
    'Acceptor',
    'SimpleFunctionAcceptor',
    'UniformAcceptor',
    'StochasticAcceptor',
    # temperature scheme
    'scheme_acceptance_rate',
    'scheme_polynomial_decay',
    'scheme_exponential_decay',
    'scheme_daly',
    'scheme_ess',
    'scheme_friel_pettitt',
    # pdf max eval
    'pdf_max_take_from_kernel',
    'pdf_max_take_max_found',
]
