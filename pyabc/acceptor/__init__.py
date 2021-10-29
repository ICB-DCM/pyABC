"""
Acceptors
=========

Acceptors handle the acceptance step.

"""


from .acceptor import (
    Acceptor,
    AcceptorResult,
    SimpleFunctionAcceptor,
    StochasticAcceptor,
    UniformAcceptor,
)
from .pdf_norm import ScaledPDFNorm, pdf_norm_from_kernel, pdf_norm_max_found
