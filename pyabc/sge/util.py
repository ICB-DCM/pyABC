import os
import subprocess  # noqa: S404


def sge_available():
    """
    Makes a simple heuristic test to check if the SGE is available on
    the machine.
    It tries to exectute the ``qstat`` command. In case it is found,
    it is assumed
    that the SGE is available.

    Returns
    -------

    available: bool
        Whether SGE is available or not.
    """
    try:
        subprocess.run("qstat", stdout=subprocess.PIPE)  # noqa: S607,S603
        return True
    except FileNotFoundError:
        return False


def nr_cores_available() -> int:
    """
    Determine the number of available cores in a manner which is safer
    for cluster environments than counting the number of CPUs the machine has.
    The CPU count might not be adequate as a job on a cluster might not
    have access to all the cores present on the cluster node on which it
    executes due to resource restrictions, such as for example done by SGE,
    SLURM etc.

    The followin heuristic scheme is used to get the available number of cores:

    1. Tries to determin cores form the SGE environment variable ``NSLOTS``
    2. From the environment variable ``OMP_NUM_THREADS``
    3. From the environment variable ``MKL_NUM_THREADS``
    4. from Python's ``os.cpu_count``

    Returns
    -------
    nr_cores: int
        The number of cores available.
    """
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
