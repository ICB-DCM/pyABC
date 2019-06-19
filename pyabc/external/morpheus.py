import pandas as pd
import numpy as np
import tempfile
import subprocess
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Callable, Any

from ..model import Model
from ..parameters import Parameter
from .base import ExternalModel


class MorpheusModel(ExternalModel):
    """
    Call morpheus model from PyABC.

    Parameters
    ----------

    morpheus_file: str
        The XML file containing the morpheus model.
    exec_name: str, optional
        The path to the morpheus executable. If None given,
        'morpheus' is used.
    suffix, prefix: str, optional (default: None, 'morpheus_model_')
        Suffix and prefix to use for the temporary folders created.
    dir: str, optional (default: None)
        Directory to put the temporary folders into. The default is
        the system's temporary files location. Note that these files
        are usually deleted upon system shutdown.
    name: str, optional (default: None)
        A name that can be used to identify the model, as it is
        saved to db. If None is passed, the model_file name is used.
    output: Callable[str, Any], optional (default: output_dict)
        What kind of output the model sample function shall give.
        Pre-defined are output_dir, output_dataframe, output_dict.
    """
    def __init__(self,
                 model_file: str,
                 exec_name: str = "morpheus",
                 suffix: str = None,
                 prefix: str = "morpheus_model_",
                 dir: str = None,
                 name: str = None,
                 output: Callable[[str], Any] = None):
        if name is None:
            name = model_file
        super().__init__(
            exec_name=exec_name,
            model_file=model_file,
            suffix=suffix, prefix=prefix, dir=dir,
            name=name)
        if output is None:
            output = output_dict
        self.output = output

    def __str__(self):
        s = f"MorpheusModel {{\n" \
            f"\texec_name:\t{self.exec_name}\n" \
            f"\tmodel_file:\t{self.model_file}\n" \
            f"\tname:\t{self.name}\n" \
            f"\toutput:\t{self.output.__name__}\n" \
            f"}}"
        return s

    def __repr__(self):
        return self.__str__()

    def sample(self, pars: Parameter):
        """
        The sample function. This function is used in ABCSMC to
        simulate data for given parameters `pars`.
        """
        # create a new folder
        dir_ = tempfile.mkdtemp(
            suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        file_ = os.path.join(dir_, "model.xml")

        # write new file with parameter modifications
        # TODO use morpheus -[KEY]=[VAL]
        self.write_modified_model_file(file_, pars)

        # create command
        cmd = f"{self.exec_name} -file={file_} -outdir={dir_}"

        # call the model
        try:
            devnull = open(os.devnull, 'w')
            subprocess.check_call(
                cmd, shell=True, stdout=devnull, stderr=devnull)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Simulation error: {e.returncode} (err: {e.output})")

        return self.output(dir=dir_)

    def write_modified_model_file(self, file_, pars):
        """
        Write a modified version of the morpheus xml file to the target
        directory.
        """
        # TODO: cache, check for validity, and allow for specific mapping
        # read xml file
        tree = ET.parse(self.model_file)
        root = tree.getroot()
        # fill in parameters
        for key, val in pars.items():
            # first try global parameters
            node = root.find(
                f"./Global/Constant[@symbol='{key}']")
            if node is None:
                # try cell type parameters
                node = root.find(
                    f"./CellTypes/CellType/System/Constant[@symbol='{key}']")
            # update value
            node.set("value", str(val))
        # write to new file
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
