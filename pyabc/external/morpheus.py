import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import Callable, Any
from ..parameters import Parameter
from .base import ExternalModel
import fitmulticell.sumstat.hexagonal_cluster_sumstat as chx
import fitmulticell.sumstat.cell_types_cout as css


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
                 sumstatfunc_list: list,
                 argument_list: list,
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
        self.sumstatfunc_list = sumstatfunc_list
        self.argument_list = argument_list

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
        loc = self.eh.create_loc()
        file_ = os.path.join(loc, "model.xml")

        # write new file with parameter modifications
        self.write_modified_model_file(file_, pars)

        # create command
        cmd = self.eh.create_executable(loc)
        cmd = cmd + f" -file={file_} -outdir={loc}"

        # call the model
        self.eh.run(cmd=cmd, loc="")
        self.argument_list[0] = loc
        # call SummeryStatistic function
        result_dict = self.sumstatlib_alltp(loc)
        return result_dict

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

    def sumstatlib_tp(self, dir):
        result_dict = {'file': dir}
        cluster_ss_obj = chx.ClusterSumstat
        cell_ss_obj = css.CellSumstat
        logger_df = read_morpheus_log_file(self.argument_list[0])
        self.argument_list[0] = logger_df
        for i in self.sumstatfunc_list:
            try:
                func = getattr(cluster_ss_obj, i)
                result_datatype = func(cluster_ss_obj, *self.argument_list)
                # check if result is int
                if isinstance(result_datatype, int):
                    result_dict[i] = result_datatype
                # check if result is dict of dict
                elif isinstance(result_datatype.values(), dict):
                    for key, value in result_datatype.items():
                        for key2, value2 in value:
                            result_dict[i + "_" + str(key) + "_" +
                                        str(key2)] = np.array(value2)
                # check if result is dict
                elif isinstance(result_datatype, dict):
                    for key, value in result_datatype.items():
                        result_dict[i + "_" + str(key)] = value
            except:
                func = getattr(cell_ss_obj, i)
                result_datatype = func(cell_ss_obj, *self.argument_list)
                # check if result is int
                if isinstance(result_datatype, int):
                    result_dict[i] = result_datatype
                # check if result is dict of dict
                elif isinstance(result_datatype.values(), dict):
                    for key, value in result_datatype.items():
                        for key2, value2 in value:
                            result_dict[i + "_" + str(key) + "_"
                                        + str(key2)] = np.array(value2)
                # check if result is dict
                elif isinstance(result_datatype, dict):
                    for key, value in result_datatype.items():
                        result_dict[i + "_" + str(key)] = value
        return result_dict

    def sumstatlib_alltp(self, dir):
        result_dict = {'file': dir}
        logger_df = read_morpheus_log_file(dir)
        cluster_ss_obj = chx.ClusterSumstat
        cell_ss_obj = css.CellSumstat
        logger_df = read_morpheus_log_file(self.argument_list[0])
        self.argument_list[0] = logger_df
        for i in self.sumstatfunc_list:
            try:
                func = getattr(cluster_ss_obj, i)
                result_datatype = func(cluster_ss_obj, *self.argument_list)
                if isinstance(result_datatype, int):
                    result_dict[i] = result_datatype
                elif set(map(type, result_datatype.values())) == {list}:
                    for key, value_list in result_datatype.items():
                        for list_member in value_list:
                            for key2, value2 in list_member.items():
                                result_dict[i + "_" + str(key) + "_" +
                                            str(key2)] = value2
                elif isinstance(result_datatype, dict):
                    for key, value in result_datatype.items():
                        result_dict[i + "_" + str(key)] = value
            except:
                func = getattr(cell_ss_obj, i)
                result_datatype = func(cell_ss_obj, *self.argument_list)
                if isinstance(result_datatype, int):
                    result_dict[i] = result_datatype
                elif set(map(type, result_datatype.values())) == {list}:
                    for key, value_list in result_datatype.items():
                        for list_member in value_list:
                            for key2, value2 in list_member.items():
                                result_dict[i + "_" + str(key) + "_" +
                                            str(key2)] = value2
                elif isinstance(result_datatype, dict):
                    for key, value in result_datatype.items():
                        result_dict[i + "_" + str(key)] = value
        return result_dict


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
