import pandas as pd
import numpy as np
import tempfile
import subprocess
import os

from ..model import Model


class ExternalModel(Model):
    """
    Interface to a model that is called via an external simulator.

    Parameters are passed to the model as named command line arguments
    in the form
    {script_name} {model_file} {arg1=value1} {arg2=value2} ... {file=filename}
    where {filename} is the name of a temporary file that was created to
    store the script output.

    In the __call__ method, the filename is returned and can be used by the
    FileSumStats or similar classes to read in the data.

    .. note::
        The generated temporary files are not automatically deleted, unless
        by the system e.g. in the /tmp directory upon restart.
    """

    def __init__(self, script_name: str, model_file: str,
                 create_dir: bool = True,
                 suffix: str = None, prefix: str = "modelsim_",
                 dir: str = None,
                 name: str = "ExternalModel"):
        """
        Initialize the model.

        Parameters
        ----------
        script_name: str
            Name of the script, e.g. bash, java or Rscript.
        model_file: str
            Path to the model to be called, e.g. a
            .sh, .java or .r file.
        create_dir: bool, optional (default = True)
            Whether the function should create a temporary directory.
            If False, only one temporary file is created.
        suffix, prefix, dir: str, optional (default = None)
            Specify suffix, prefix, or base directory for the created
            temporary files.
        name: str, optional (default = "ExternalModel")
            As in pyabc.Model.name.
        """
        super().__init__(name=name)
        self.script_name = script_name
        self.model_file = model_file
        self.create_dir = create_dir
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir

    def __call__(self, pars):
        args = []
        for key, val in pars.items():
            args.append(f"{key}={val} ")
        if self.create_dir:
            file_fun = tempfile.mkdtemp
        else:
            file_fun = tempfile.mkstemp
        loc_ = file_fun(
            suffix=self.suffix, prefix=self.prefix, dir=self.dir)[1]
        args.append(f"file={loc_}")
        subprocess.run([self.script_name, self.model_file, *args])
        return loc_

    def sample(self, pars):
        return self(pars)


class ExternalSumStat:
    """
    {script_name} {sumstat_file} {model_output_file=model_output_fild}
    {file=file}
    """

    def __init__(self, script_name, sumstat_file,
                 suffix=None, prefix="sumstat_", dir=None):
        self.script_name = script_name
        self.sumstat_file = sumstat_file
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir

    def __call__(self, model_output_file):
        args = []
        args.append(f"model_output_file={model_output_file}")
        file_ = tempfile.mkstemp(
            suffix=self.suffix, prefix=self.prefix, dir=self.dir)[1]
        args.append(f"file={file_}")
        ret = subprocess.run([self.script_name, self.sumstat_file, *args])
        return {'file': file_, 'returncode': ret.returncode}


class ExternalDistance:
    """
    Use script and sumstat output files to compute the distance.

    The distance is written to a file, which is then read in (it must only
    contain a single float number).
    """

    def __init__(self, script_name, distance_file,
                 suffix=None, prefix="dist_", dir=None):
        self.script_name = script_name
        self.distance_file = distance_file
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir

    def __call__(self, sumstat_0, sumstat_1):
        # check if external script failed
        if sumstat_0['returncode'] or sumstat_1['returncode']:
            return np.nan
        args = [f"sumstat_0_file={sumstat_0['file']}",
                f"sumstat_1_file={sumstat_1['file']}"]
        file_ = tempfile.mkstemp(
            suffix=self.suffix, prefix=self.prefix, dir=self.dir)[1]
        args.append(f"file={file_}")
        subprocess.run([self.script_name, self.distance_file, *args])
        # read in distance
        with open(file_, 'rb') as f:
            distance = float(f.read())
        os.remove(file_)
        return distance


class FileIdSumStat:
    def __init__(self, sep):
        self.sep = sep

    def __call__(self, model_output_file):
        df = pd.read_csv(model_output_file, sep=self.sep)
        dct = dict()
        for col in df.columns:
            dct[col] = np.array(df[col])
        # TODO: Recognize empty columns (reduce vector size)
        # and reduce 1-dim vectors to scalars
        return dct
