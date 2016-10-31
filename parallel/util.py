import os
import subprocess


def sge_available():
    """
    Makes a simple heuristic test to check if the SGE is available on the machine.
    It tries to exectute the ``qstat`` command. In case it is found, it is assumed
    that the SGE is available.

    Returns
    -------

    available: bool
        Whether SGE is available or not.
    """
    try:
        subprocess.run("qstat", stdout=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def nr_cores_available():
    try:
        return int(os.environ['NSLOTS'])
    except KeyError:
        pass
    try:
        return int(os.environ['OMP_NUM_THREADS'])
    except KeyError:
        pass
    try:
        return int(os.environ['MKL_NUM_THREADS'])
    except KeyError:
        pass
    return os.cpu_count()
