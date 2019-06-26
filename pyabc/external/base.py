import numpy as np
import tempfile
import subprocess
import os
from typing import List

from ..model import Model


class ExternalHandler:
    """
    Handler for calls to external scripts.

    This is a convenience class for bundling repeated functionality.
    """

    def __init__(self,
                 executable: str, file: str,
                 fixed_args: List = None,
                 create_folder: bool = False,
                 suffix: str = None, prefix: str = None, dir: str = None):
        """
        Parameters
        ----------
        executable: str
            Name of the executable to call, e.g. bash, java or Rscript.
        file: str
            Path to the file to be executed, e.g. a
            .sh, .java or .r file, or also a .xml file depending on the
            executable.
        fixed_args: str, optional (default = None)
            Argument string to use every time.
        create_folder: bool, optional (default = True)
            Whether the function should create a temporary directory.
            If False, only one temporary file is created.
        suffix, prefix, dir: str, optional (default = None)
            Specify suffix, prefix, or base directory for the created
            temporary files.
        """
        self.executable = executable
        self.file = file
        if fixed_args is None:
            fixed_args = []
        self.fixed_args = fixed_args
        self.create_folder = create_folder
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir

    def create_loc(self):
        """
        Create temporary file or folder.

        Returns
        -------
        loc: str
            Path of the created file or folder.
        """
        if self.create_folder:
            return tempfile.mkdtemp(
                suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        else:
            return tempfile.mkstemp(
                suffix=self.suffix, prefix=self.prefix, dir=self.dir)[1]

    def run(self, args):
        """
        Run the script for the given arguments.
        """
        # create target on file system
        loc = self.create_loc()
        # redirect output
        devnull = open(os.devnull, 'w')
        # call
        status = subprocess.run(
            [self.executable, self.file, *self.fixed_args, *args,
             f'target={loc}'],
            stdout=devnull, stderr=devnull)
        # return location and call's return status
        return {'loc': loc, 'returncode': status.returncode}


class ExternalModel(Model):
    """
    Interface to a model that is called via an external simulator.

    Parameters are passed to the model as named command line arguments
    in the form:
        {executable} {file} {par1}={val1} {par2}={val2} ... target={loc}
    Here, {file} is the script that performs the model simulation, and {loc}
    is the name of a temporary file or folder that was created to
    store the simulated data.

    .. note::
        The generated temporary files are not automatically deleted, unless
        by the system e.g. in the /tmp directory upon restart.
    """

    def __init__(self, executable: str, file: str,
                 fixed_args: List = None,
                 create_folder: bool = False,
                 suffix: str = None, prefix: str = "modelsim_",
                 dir: str = None,
                 name: str = "ExternalModel"):
        """
        Initialize the model.

        Parameters
        ----------
        name: str, optional (default = "ExternalModel")
            As in pyabc.Model.name.

        All other parameters as in ExternalHandler.
        """
        super().__init__(name=name)
        self.eh = ExternalHandler(
            executable=executable, file=file,
            fixed_args=fixed_args,
            create_folder=create_folder,
            suffix=suffix, prefix=prefix, dir=dir)

    def __call__(self, pars):
        args = []
        for key, val in pars.items():
            args.append(f"{key}={val} ")
        return self.eh.run(args)

    def sample(self, pars):
        return self(pars)


class ExternalSumStat:
    """
    Interface to an external calculator that takes the simulated model output
    and writes to file the summary statistics.

    Format:
        {executable} {file} model_output={model_output} target={loc}
    Here, {file} is the path to the summary statistics computation script,
    {model_output} is the path to the previously generated model output, and
    {loc} is the destination to write te summary statistics to.
    """

    def __init__(self, executable: str, file: str,
                 fixed_args: List = None,
                 create_folder: bool = False,
                 suffix: str = None, prefix: str = "sumstat_",
                 dir: str = None):
        self.eh = ExternalHandler(
            executable=executable, file=file,
            fixed_args=fixed_args,
            create_folder=create_folder,
            suffix=suffix, prefix=prefix, dir=dir)

    def __call__(self, model_output):
        """
        Create summary statistics from the `model_output` generated
        by the model.
        """
        args = [f"model_output={model_output['loc']}"]
        return self.eh.run(args=args)


class ExternalDistance:
    """
    Use script and sumstat output files to compute the distance.

    Format:
        {executable} {file} sumstat_0={sumstat_0} sumstat_1={sumstat_1}
        target={loc}

    The distance is written to a file, which is then read in (it must only
    contain a single float number).
    """

    def __init__(self, executable: str, file: str,
                 fixed_args: List = None,
                 suffix: str = None, prefix: str = "dist_",
                 dir: str = None):
        self.eh = ExternalHandler(
            executable=executable, file=file,
            fixed_args=fixed_args,
            create_folder=False,
            suffix=suffix, prefix=prefix, dir=dir)

    def __call__(self, sumstat_0, sumstat_1):
        # check if external script failed
        if sumstat_0['returncode'] or sumstat_1['returncode']:
            return np.nan
        args = [f"sumstat_0={sumstat_0['loc']}",
                f"sumstat_1={sumstat_1['loc']}"]
        ret = self.eh.run(args)
        # read in distance
        with open(ret['loc'], 'rb') as f:
            distance = float(f.read())
        os.remove(ret['loc'])
        return distance


def create_sum_stat(loc: str = '', returncode: int = 0):
    """
    Create a summary statistics dictionary, as returned by the
    `ExternalModel`.

    Can be used to encode the measured summary statistics, or
    also create a dummy summary statistic.

    Parameters
    ----------
    loc: str, optional (default = '')
        Location of the summary statistics file or folder.
    returncode: int, optional (default = 0)
        Defaults to 0, indicating correct execution. Should usually
        not be changed.

    Returns
    -------
    A dictionary with keys 'loc' and 'returncode' of the given
    parameters.
    """
    return {'loc': loc, 'returncode': returncode}
