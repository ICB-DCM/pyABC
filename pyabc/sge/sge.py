import inspect
import os
import pickle
import shutil
import subprocess
import tempfile
import time
import sys
import cloudpickle
from .config import make_config
from .execution_contexts import DefaultContext
from .db import job_db_factory
from .util import sge_available
import warnings


class SGESignatureMismatchException(Exception):
    pass


class SGE:
    """Map a function to be executed on an SGE cluster environment
    Reads a config file (if it exists) in you home directory
    which should look as the default
    in sge.config.

    The mapper reads commonly used parameters from a configuration file
    stored in ``~/.parallel``
    An example configuration file could look as follows:


    .. code-block:: bash

        #~/.parallel
        [DIRECTORIES]
        TMP=/tmp

        [BROKER]
        # The value of TYPE can be SQLITE or REDIS
        TYPE=REDIS

        [SGE]
        QUEUE=p.openmp
        PARALLEL_ENVIRONMENT=openmp
        PRIORITY=-500

        [REDIS]
        HOST=127.0.0.1


    Parameters
    ----------

    tmp_directory: str or None
        Directory where temporary job pickle files are stored
        If set to None a tmp directory is read from the ''~/.parallel''
        configuration file.
        It this file does not exist a tmp directory within the user home
        directory is created.

    memory: str, optional (default = '3G')
        Ram requested by each job, e.g. '10G'.

    time_h: int (default = 100)
        Job run time in hours.

    python_executable_path: str or None
        The python interpreter which executes the jobs.
        If set to None, the currently executing interpreter is used as
        returned by ``sys.executable``.

    sge_error_file: str or None
        File to which stderr messages from workers are stored.
        If set to None, a file within the tmp_directory is used.

    sge_output_file: str or None
        File to which stdout messages from workers are stored
        If set to None, a file within the tmp_directory is used.

    parallel_environment: str, optional (default = 'map')
        The SGE environment. (This is what is passed to the -pe option
        in the qsub script).

    name: str
        A name for the job.

    queue: str
        The SGE queue.

    priority: int, optional.
        SGE job priority. A value between -1000 and 0.
        Note that a priority of 0 automatically enables the reservation flag.

    num_threads: int, optional (default = 1)
        Number of threads for each worker.
        This also sets the environment variable MKL_NUM_THREADS,
        OMP_NUM_THREADS to the
        specified number to handle jobs which use OpenMP etc. correctly.

    execution_context: \
    :class:`DefaultContext \
        <pyabc.sge.execution_contexts.DefaultContext>`,\
    :class:`ProfilingContext \
        <pyabc.sge.execution_contexts.ProfilingContext>`,\
    :class:`NamedPrinter \
        <pyabc.sge.execution_contexts.NamedPrinter>`

        Any context manager can be passed here.
        The ``__enter__`` method is called before evaluating the function on
        the cluster.
        The ``__exit__`` method directly after the function run finished.

    chunk_size: int, optional (default = 1)
        nr of tasks executed within one job.

            .. warning::

                If ``chunk_size`` is larger than 1, this can have
                side effects
                as all the jobs within one chunk are executed within the python
                process.


    Returns
    -------
    sge: SGE
        The configured sge mapper.
    """

    def __init__(self, tmp_directory: str = None, memory: str = '3G',
                 time_h: int = 100,
                 python_executable_path: str = None,
                 sge_error_file: str = None,
                 sge_output_file: str = None,
                 parallel_environment=None, name="map",
                 queue=None, priority=None, num_threads: int = 1,
                 execution_context=DefaultContext, chunk_size=1):

        # simple assignments
        self.memory = memory
        self.time_h = time_h

        self.config = make_config()

        if parallel_environment is not None:
            self.config["SGE"]["PARALLEL_ENVIRONMENT"] = parallel_environment
        if queue is not None:
            self.config["SGE"]["QUEUE"] = queue
        if tmp_directory is not None:
            self.config["DIRECTORIES"]["TMP"] = tmp_directory
        if priority is not None:
            self.config["SGE"]["PRIORITY"] = str(priority)

        try:
            os.makedirs(self.config["DIRECTORIES"]["TMP"])
        except FileExistsError:
            pass

        self.time = str(time_h) + ":00:00"

        self.job_name = name
        if self.config["SGE"]["PRIORITY"] == "0":
            warnings.warn("Priority set to 0. "
                          "This enables the reservation flag.")
        self.num_threads = num_threads
        self.execution_context = execution_context
        self.chunk_size = chunk_size

        if chunk_size != 1:
            warnings.warn("Chunk size != 1. "
                          "This can potentially have bad side effect.")

        if not sge_available():
            print("Warning: Could not find SGE installation.", file=sys.stderr)

        # python interpreter which executes the jobs
        if not isinstance(time_h, int):
            raise Exception('Time should be an integer hour')
        if python_executable_path is None:
            python_executable_path = sys.executable
        self.python_executable_path = python_executable_path

        # sge stderr
        if sge_error_file is not None:
            self.sge_error_file = sge_error_file
        else:
            self.sge_error_file = os.path.join(
                self.config["DIRECTORIES"]["TMP"], "sge_errors.txt")

        # sge stdout
        if sge_output_file is not None:
            self.sge_output_file = sge_output_file
        else:
            self.sge_output_file = os.path.join(
                self.config["DIRECTORIES"]["TMP"], "sge_output.txt")

    def __repr__(self):
        return ("<SGE memory={} time={} priority={} num_threads={} "
                "chunk_size={} tmp_dir={} python_executable={}>"
                .format(self.memory, self.time, self.config["SGE"]["PRIORITY"],
                        self.num_threads, self.chunk_size,
                        self.config["DIRECTORIES"]["TMP"],
                        self.python_executable_path))

    @staticmethod
    def _validate_function_arguments(function, array):
        """

        Parameters
        ----------
        function: function to be mapped
        array: array of values

        Raises and Exception if arguments do not match
        This prevents unnecessary Job submission and raises an error earlier

        """
        signature = inspect.signature(function)
        try:
            for argument in array:
                signature.bind(argument)
        except TypeError as err:
            raise SGESignatureMismatchException(
                "Your jobs were not submitted as the function could not be "
                "applied to the arguments.") from err

    def map(self, function, array):
        """
        Da what map(function, array) would do, but do it
        via a array job on the SGE by pickling objects, storing
        them in a temporary folder, submitting them to SGE and
        then reading and returning the results.

        Parameters
        ----------

        function: Callable

        array: iterable

        Returns
        -------

        result_list: list
            List of results of function application.
            This list can also contain ``Exception`` objects.
        """
        array = list(array)

        self._validate_function_arguments(function, array)
        tmp_dir = tempfile.mkdtemp(prefix="", suffix='_SGE_job',
                                   dir=self.config["DIRECTORIES"]["TMP"])

        # jobs directory
        jobs_dir = os.path.join(tmp_dir, 'jobs')
        os.mkdir(jobs_dir)

        # create results directory
        os.mkdir(os.path.join(tmp_dir, 'results'))

        # store the function
        with open(os.path.join(tmp_dir, 'function.pickle'), 'wb') as my_file:
            cloudpickle.dump(function, my_file)

        # store execution context
        with open(os.path.join(tmp_dir, 'ExecutionContext.pickle'), 'wb') \
                as my_file:
            cloudpickle.dump(self.execution_context, my_file)

        # store the array
        for task_nr, start_index in enumerate(range(0, len(array),
                                                    self.chunk_size)):
            with open(os.path.join(jobs_dir,
                                   str(task_nr + 1) + '.job'),
                      'wb') as my_file:
                cloudpickle.dump(
                    list(array[start_index:start_index + self.chunk_size]),
                    my_file)

        nr_tasks = task_nr + 1

        batch_file = self._render_batch_file(nr_tasks, tmp_dir)

        with open(os.path.join(tmp_dir, 'job.sh'), 'w') as my_file:
            my_file.write(batch_file)

        # crate job jd
        job_db = job_db_factory(tmp_dir)
        job_db.create(len(array))

        # start the job with qsub
        subprocess.run(['qsub', os.path.join(tmp_dir, 'job.sh')],
                       stdout=subprocess.PIPE)

        # wait for the tasks to be finished
        finished = False
        while not finished:
            time.sleep(5)
            for k in range(nr_tasks)[::-1]:  # check from last to first job
                # +1 offset for k b/c SGE array jobs start at 1, not at 0
                if job_db.wait_for_job(k + 1, self.time_h):
                    break
            else:
                finished = True
        # from here on all tasks are finished

        # sleep to make sure files are entirely written
        time.sleep(5)

        # make the results array
        results = []
        had_exception = False
        for task_nr in range(nr_tasks):
            try:
                my_file = open(
                    os.path.join(tmp_dir, 'results', str(task_nr + 1)
                                 + '.result'), 'rb')
                single_result = pickle.load(my_file)
                results += single_result
                my_file.close()
            except Exception as e:
                results.append(Exception('Could not load temporary '
                                         'result file:' + str(e)))
                had_exception = True

        # delete the temporary folder if there was no problem
        # and execution context does not need it
        if self.execution_context.keep_output_directory:
            pass
        elif had_exception:
            tmp_dir = tmp_dir[:-1] if tmp_dir[-1] == '/' else tmp_dir
            os.rename(tmp_dir, tmp_dir + '_with_exception')
        else:
            shutil.rmtree(tmp_dir)
            job_db.clean_up()
        return results

    def _render_batch_file(self, nr_tasks, tmp_dir):
        # create the file to be submitted to SGE via qsub
        # for array jobs the ressource request are per task!
        try:
            pythonpath = os.environ['PYTHONPATH']
        except KeyError:
            pythonpath = ""

        batch_file = """#!/bin/bash
#$ -N {job_name}-{map_name}
#$ -S /bin/bash
#$ -q {queue}
#$ -pe {parallel_environment} {num_threads}
#$ -l h_vmem={RAM}
#$ -l h_rt={time}
#$ -p {priority}
#$ -R {reservation}
#$ -V
#$ -t 1-{nr_elements}
#$ -e {sge_error_file}
#$ -o {sge_output_file}
cd "{current_working_directory}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export PYTHONPATH={pythonpath}
{executable} -m pyabc.sge.execute_sge_array_job {tmp_dir} $SGE_TASK_ID
{executable} -m pyabc.sge.cleanup_sge_array_job {tmp_dir} $SGE_TASK_ID
""".format(tmp_dir=tmp_dir, nr_elements=nr_tasks,
           current_working_directory=os.getcwd(),
           RAM=self.memory, time=self.time,
           job_name=self.job_name,
           executable=self.python_executable_path,
           sge_error_file=self.sge_error_file,
           sge_output_file=self.sge_output_file,
           priority=self.config["SGE"]["PRIORITY"],
           parallel_environment=self.config["SGE"]["PARALLEL_ENVIRONMENT"],
           queue=self.config["SGE"]["QUEUE"],
           map_name=os.path.split(tmp_dir)[-1],
           num_threads=self.num_threads,
           pythonpath=pythonpath,
           reservation="y" if self.config["SGE"]["PRIORITY"] == "0" else "n")
        return batch_file
