import pandas as pd
import numpy as np
import tempfile
import subprocess
import os
import shutil
import xml.etree.ElementTree as ET

from ..model import Model
from ..parameters import Parameter
from .script import ExternalModel


class MorpheusModel(ExternalModel):
    """
    Call morpheus model from PyABC.

    Parameters
    ----------

    morpheus_file:
        The XML file containing the morpheus model.
    """
    def __init__(self, model_file,
                 suffix=None, prefix="morpheus_model__", dir=None,
                 name="MorpheusModel"):
        super().__init__(
            script_name="morpheus",
            model_file=model_file,
            suffix=suffix, prefix=prefix, dir=dir,
            name=name)

    def sample(self, pars: Parameter):
        # create a new folder
        dir_ = tempfile.mkdtemp(
            suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        file_ = os.path.join(dir_, "model.xml")

        # write new file with parameter modifications
        self.write_modified_model_file(file_, pars)

        # create command
        cmd = [self.script_name, f"-file={file_}"]

        # call the model
        cwd = os.getcwd()  # change working directory
        os.chdir(dir_)
        subprocess.run(cmd)
        os.chdir(cwd)  # undo change

        # return the created directory
        return {'dir': dir_}

    def write_modified_model_file(self, file_, pars):
        # read xml file
        tree = ET.parse(self.model_file)
        root = tree.getroot()
        # fill in parameters
        for key, val in pars.items():
            node = root.find(
                f"./CellTypes/CellType/System/Constant[@symbol='{key}']")
            node.set("value", str(val))
        # write to new file
        tree.write(file_)


class MorpheusData:
    """
    Read in data from morpheus folder.
    """

    def __call__(self, model_output):
        data_file = os.path.join(model_output['dir'], "logger.csv")
        df = pd.read_csv(data_file, sep="\t")
        return df
