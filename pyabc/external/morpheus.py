import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import Callable, Any

from ..parameters import Parameter
from .base import ExternalModel


class MorpheusModel(ExternalModel):
    """
    Call morpheus model from PyABC.

    Parameters
    ----------

    model_file: str
        The XML file containing the morpheus model.
    par_map: dict
        A dictionary from str to str, the keys being the parameter ids
        to be used in pyabc, and the values xpaths in the `morpheus_file`.
    executable: str, optional
        The path to the morpheus executable. If None given,
        'morpheus' is used.
    suffix, prefix: str, optional (default: None, 'morpheus_model_')
        Suffix and prefix to use for the temporary folders created.
    dir: str, optional (default: None)
        Directory to put the temporary folders into. The default is
        the system's temporary files location. Note that these files
        are usually deleted upon system shutdown.
    show_stdout, show_stderr: bool, optional (default = False, True)
        Whether to show or hide the stdout and stderr streams.
    raise_on_error: bool, optional (default = False)
        Whether to raise on an error in the model execution, or
        just continue.
    name: str, optional (default: None)
        A name that can be used to identify the model, as it is
        saved to db. If None is passed, the model_file name is used.
    output: Callable[str, Any], optional (default: output_dict)
        What kind of output the model sample function shall give.
        Pre-defined are output_dir, output_dataframe, output_dict.
    """

    def __init__(self,
                 model_file: str,
                 par_map: dict,
                 executable: str = "morpheus",
                 suffix: str = None,
                 prefix: str = "morpheus_model_",
                 dir: str = None,
                 show_stdout: bool = False,
                 show_stderr: bool = True,
                 raise_on_error: bool = False,
                 name: str = None,
                 output: Callable[[str], Any] = None):
        if name is None:
            name = model_file
        super().__init__(
            executable=executable,
            file=model_file,
            fixed_args=None,
            create_folder=True,
            suffix=suffix, prefix=prefix, dir=dir,
            show_stdout=show_stdout,
            show_stderr=show_stderr,
            raise_on_error=raise_on_error,
            name=name)
        self.par_map = par_map
        if output is None:
            output = output_dict
        self.output = output

    def __str__(self):
        s = f"MorpheusModel {{\n" \
            f"\texecutable: {self.eh.executable}\n" \
            f"\tfile      : {self.eh.file}\n" \
            f"\tname      : {self.name}\n" \
            f"\toutput    : {self.output.__name__}\n" \
            f"}}"
        return s

    def __repr__(self):
        return self.__str__()

    def __call__(self, pars: Parameter):
        """
        This function is used in ABCSMC (or rather the sample() function,
        which redirects here) to simulate data for given parameters `pars`.
        """
        # create target on file system
        dir_ = self.eh.create_loc()
        file_ = os.path.join(dir_, "model.xml")

        # write new file with parameter modifications
        self.write_modified_model_file(file_, pars)

        # create command
        cmd = f"{self.eh.executable} -file={file_} -outdir={dir_}"

        # call the model
        self.eh.run(cmd=cmd, loc="")

        return self.output(dir=dir_)

    def write_modified_model_file(self, file_, pars):
        """
        Write a modified version of the morpheus xml file to the target
        directory.
        """
        tree = ET.parse(self.eh.file)
        root = tree.getroot()
        for key, val in pars.items():
            xpath = self.par_map[key]
            node = root.find(xpath)
            node.set('value', str(val))
        tree.write(file_)


def output_dir(dir):
    """Output the directory."""
    return {'dir': dir}


def output_dataframe(dir):
    """Output as pandas.DataFrame."""
    df = read_morpheus_log_file(dir)
    return {'dataframe': df}


def output_dict(dir):
    """Output as dictionary with numpy.ndarray's."""
    df = read_morpheus_log_file(dir)
    # convert to dict
    dct = df.to_dict(orient='list')
    # use numpy arrays
    for key, val in dct.items():
        dct[key] = np.array(val)
    return dct


def read_morpheus_log_file(dir):
    """Read in the morpheus logging file inside directory `dir`."""
    data_file = os.path.join(dir, "logger.csv")
    data_frame = pd.read_csv(data_file, sep="\t")
    return data_frame
