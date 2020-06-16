"""
Acceptors
=========

Acceptors handle the acceptance step.

"""


from .acceptor import (
    AcceptorResult,
    Acceptor,
    SimpleFunctionAcceptor,
    UniformAcceptor,
    StochasticAcceptor,
)
from .pdf_norm import (
    pdf_norm_from_kernel,
    pdf_norm_max_found,
    ScaledPDFNorm,
)
