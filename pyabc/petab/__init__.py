"""
PEtab
=====

Problem definitions in the PEtab format (https://petab.rtfd.io).
"""
import warnings

from .amici import AmiciPetabImporter

try:
    import petab
except ImportError:
    warnings.warn(
        "PEtab import requires an installation of petab "
        "(https://github.com/PEtab-dev/PEtab). "
        "Install via `pip3 install petab`.",
        stacklevel=1,
    )
