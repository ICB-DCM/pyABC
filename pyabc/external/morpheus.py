import pandas as pd
import numpy as np
import tempfile
import subprocess
import os
import shutil
import xml.etree.ElementTree as ET

from ..model import Model
from ..parameters import Parameter
from .base import ExternalModel


class MorpheusModel(ExternalModel):
    """
    Call morpheus model from PyABC.

    Parameters
    ----------

    morpheus_file:
        The XML file containing the morpheus model.
    """
    def __init__(self,
                 model_file: str,
                 exec_name: str = "morpheus",
                 suffix: str = None,
                 prefix: str = "morpheus_model__",
                 dir: str = None,
                 name: str = "MorpheusModel",
                 output: str = 'dir'):
        super().__init__(
            exec_name=exec_name,
            model_file=model_file,
            suffix=suffix, prefix=prefix, dir=dir,
            name=name)
        self.output = output

    def __str__(self):
        s = f"MorpheusModel {{\n" \
            f"\texec_name:\t{self.exec_name}" \
            f"\tmodel_file:\t{self.model_file}" \
            f"\tname:\t{self.name}" \
            f"\toutput:\t{self.output}" \
            f"}}"
        return s

    def __repr__(self):
        return self.__str__()

    def sample(self, pars: Parameter):
        # create a new folder
        dir_ = tempfile.mkdtemp(
            suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        file_ = os.path.join(dir_, "model.xml")

        # write new file with parameter modifications
        # TODO use morpheus -[KEY]=[VAL]
        self.write_modified_model_file(file_, pars)

        # create command
        cmd = f"{self.exec_name} -file={file_}"

        # call the model
        # change working directoy (TODO use morpheus target dir)
        cwd = os.getcwd()
        os.chdir(dir_)
        try:
            devnull = open(os.devnull, 'w')
            subprocess.check_call(
                cmd, shell=True, stdout=devnull, stderr=devnull)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Simulation error: {e.returncode} (err: {e.output})")
        os.chdir(cwd)  # undo change

        return self.create_output(dir=dir_)

    def write_modified_model_file(self, file_, pars):
        """
        Write a modified version of the morpheus xml file to the target
        directory.
        """
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

    def create_output(self, dir):
        """
        Create custom output from morpheus simulation.
        """
        out = {}
        if 'dir' in self.output:
            out['dir'] = dir
        elif 'dataframe' in self.output:
            data_file = os.path.join(dir, "logger.csv")
            df = pd.read_csv(data_file, sep="\t")
            out['data'] = df
            # TODO tidy up output
        return out
