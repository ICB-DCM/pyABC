import numpy as np
import tempfile
import subprocess  # noqa: S404
import os
from typing import List
import logging

from ..model import Model
from ..parameters import Parameter


logger = logging.getLogger("External")


class ExternalHandler:
    """
    Handler for calls to external scripts.

    This is a convenience class for bundling repeated functionality.
    """

    def __init__(self,
                 executable: str, file: str = None,
                 fixed_args: List = None,
                 create_folder: bool = False,
                 suffix: str = None, prefix: str = None, dir: str = None,
                 show_stdout: bool = False,
                 show_stderr: bool = True,
                 raise_on_error: bool = False):
        """
        Parameters
        ----------
        executable: str
            Name of the executable to call, e.g. bash, java or Rscript.
            The executable may be parameterized, e.g. appearances of {loc}
            in the string are replaced at runtime by the location of the
            output.
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
        show_stdout, show_stderr: bool, optional (default: False, True)
            Whether to show or hide the stdout and stderr streams.
        raise_on_error: bool, optional (default = False)
            Whether to raise when an error in the execution of the external
            script occurs, or just continue.
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
        self.show_stdout = show_stdout
        self.show_stderr = show_stderr
        self.raise_on_error = raise_on_error

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

    def create_executable(self, loc):
        """
        Parse and return executable.

        Replaces instances of {loc} by the location `loc`.
        """
        executable = self.executable.replace("{loc}", loc)
        return executable

    def run(self, args: List[str] = None, cmd: str = None, loc: str = None):
        """
        Run the script for the given arguments.

        Parameters
        ----------
        args: List[str], optional
            Arguments to pass to the external program, e.g. parameters.
        cmd: str, optional
            If this is not None, then it is assumed to contain the full
            command to be executed via the shell (then `args` is ignored).
            Be aware of possible security implications of shell injection.
        loc: str, optional
            Location for the output. If None is passed, one is created.
        """
        # create target on file system
        if loc is None:
            loc = self.create_loc()

        # redirect output
        devnull = open(os.devnull, 'w')
        stdout = stderr = {}
        if not self.show_stdout:
            stdout = {'stdout': devnull}
        if not self.show_stderr:
            stderr = {'stderr': devnull}

        # call
        if cmd is not None:
            status = subprocess.run(
                    cmd, shell=True, **stdout, **stderr)  # noqa: S602
        else:
            executable = self.create_executable(loc)
            status = subprocess.run(  # noqa: S603
                [executable, self.file, *self.fixed_args, *args,
                 f'target={loc}'], **stdout, **stderr)
        if status.returncode:
            msg = (f"Simulation error for arguments {args}: "
                   f"returncode {status.returncode}.")
            if self.raise_on_error:
                raise ValueError(msg)
            else:
                logger.warn(msg)

        # return location and call's return status
        return {'loc': loc, 'returncode': status.returncode}


class ExternalModel(Model):
    """
    Interface to a model that is called via an external simulator.

    Parameters are passed to the model as named command line arguments
    in the form

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
                 show_stdout: bool = False,
                 show_stderr: bool = True,
                 raise_on_error: bool = False,
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
            suffix=suffix, prefix=prefix, dir=dir,
            show_stdout=show_stdout,
            show_stderr=show_stderr,
            raise_on_error=raise_on_error)

    def __call__(self, pars: Parameter):
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
                 dir: str = None,
                 show_stdout: bool = False,
                 show_stderr: bool = True,
                 raise_on_error: bool = False):
        self.eh = ExternalHandler(
            executable=executable, file=file,
            fixed_args=fixed_args,
            create_folder=create_folder,
            suffix=suffix, prefix=prefix, dir=dir,
            show_stdout=show_stdout,
            show_stderr=show_stderr,
            raise_on_error=raise_on_error)

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
                 dir: str = None,
                 show_stdout: bool = False,
                 show_stderr: bool = True,
                 raise_on_error: bool = False):
        self.eh = ExternalHandler(
            executable=executable, file=file,
            fixed_args=fixed_args,
            create_folder=False,
            suffix=suffix, prefix=prefix, dir=dir,
            show_stdout=show_stdout,
            show_stderr=show_stderr,
            raise_on_error=raise_on_error)

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
