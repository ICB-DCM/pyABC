import pandas as pd
import numpy as np
import tempfile
import subprocess
import os
from lxml import etree as xmlm
from lxml import etree as ET

from ..model import Model


class ExternalModel(Model):
    """
    A model that is called via a command line script.

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

    def __init__(self, script_name, model_file,
                 suffix=None, prefix="modelsim_", dir=None,
                 name=None):
        """
        Initialize the model.

        Parameters
        ----------
        script_name: str
            Name of the script, e.g. bash or Rscript.
        model_file: str
            Path to the model to be called, e.g. a
            .sh or .r file.
        suffix, prefix, dir: str, optional (default = None)
            Specify suffix, prefix, or base directory for the created
            temporary files.
        """
        super().__init__(name=name)
        self.script_name = script_name
        self.model_file = model_file
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir

    def __call__(self, pars):
        args = []
        for key, val in pars.items():
            args.append(f"{key}={val} ")
        file_ = tempfile.mkstemp(
            suffix=self.suffix, prefix=self.prefix, dir=self.dir)[1]
        args.append(f"file={file_}")
        subprocess.run([self.script_name, self.model_file, *args])
        return file_

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



class MorpheusModel(Model):
    """
    Call morpheus model from PyABC.

    Parameters
    ----------

    morpheus_file:
        The XML file containing the morpheus model.
    parameter_mapping:
        Mapping from parameter names like `rate0` to the location in the
        xml file like `CellTypes.System.Constant with symbol rate0`
    """
    def __init__(self, morpheus_file, paramater_mapping=None):
        self.morpheus_file = morpheus_file

    def get_parameter_of_interest_dict(xml_path:str ,parofinterest:str) ->dict:
        """"
        read the xml file and parse the value of the parameter of interest

        parameters
        ----------
        xml_path:
            the path of the xml file
        parofinterest:
            the symbol of parameter of interest that we want to estimate

        return
        ------
        par_of_interest_dict:
            a dictionary that hold the symbol and value of the parameter of interest
        """
        #read xml file.
        tree = ET.parse(xml_path)
        root = tree.getroot()
        #read the parameter of interest form the xml file.
        for temp in root.findall("./CellTypes/CellType/System/Constant[@symbol='" + parofinterest + "']"):
            par_of_interest_dict = temp.attrib
        return par_of_interest_dict


    def get_parameter_of_interest_value( xml_path:str ,parofinterest:str) ->float:
        """"
        read the xml file and parse the value of the parameter of interest

        parameters
        ----------
        xml_path:
            the path of the xml file
        parofinterest:
            the symbol of parameter of interest that we want to estimate
        par_of_interest_value:
            the value of the parameter of interest

        """
        #read xml file.
        tree = ET.parse(xml_path)
        root = tree.getroot()
        #read the parameter of interest form the xml file.
        for temp in root.findall("./CellTypes/CellType/System/Constant[@symbol='" + parofinterest + "']"):
            par_of_interest_dict = temp.attrib
        par_of_interest_value = float(par_of_interest_dict['value'])
        return par_of_interest_value

    def set_parameter_of_interest( xml_path:str,par_of_interest_dict:list):
        # This function have a problem of encoding Unicode characters
        """"
        read the xml file and parse the value of the parameter of interest

        parameters
        ----------
        xml_path:
            the path of the xml file
        par_of_interest_dict:
            parameter of interest dictionary that hold the symbol
            and value of the new vlue of the parameter of interest
        par_of_interest_value:
            the value of the parameter of interest

        """
        #read xml file.
        tree = ET.parse(xml_path)
        root = tree.getroot()
        #read the parameter of interest form the xml file.
        par_index=root.find("./CellTypes/CellType/System/Constant[@symbol='" + par_of_interest_dict[0] + "']")
        par_index.set('value',str(par_of_interest_dict[1]))
        tree.write(xml_path)

    def sample(self,xml_path, par:dict):

        # create a new folder with a unique id
        model_folder=''
        # copy the xml into this folder
        new_file = xmlm.parse(xml_path)

        
        # in the new xml, change the parameters to `par` (I think it's the Constant Values)
        old_par_dict=self.read_parameter_of_interest_dict(xml_path, par['symbol'])

        # use some python xml editing tool

        # you have to know exactly where the parameter, say b1, will be
    
        # call the model
        subprocess.run('morpheus -file '+ new_file, shell=True)

        # the output csv file will be written into the same folder I think (logging.csv?)
        # return the folder name or the csv file name
        return model_folder


class MorpheusSumStat:

    def __call__(self, model_folder):
        modle_folder=''
        # read in the csv file as a pandas dataframe
        # extract the summary statistics (in the simplest case, do nothing)
        # what exactly you return here will depend on what data you have

        # return the summary statistics



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

