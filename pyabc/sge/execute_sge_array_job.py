import os
import pickle
import sys

import cloudpickle
from .db import job_db_factory

from pyabc.sge.execution_contexts import NamedPrinter

tmp_path = sys.argv[1]
job_nr = sys.argv[2]


# save start time to database
job_db = job_db_factory(tmp_path)
job_db.start(int(job_nr))

# load the function
with open(os.path.join(tmp_path, 'function.pickle'), 'rb') as my_file:
    function = pickle.load(my_file)

# load the array
with open(os.path.join(tmp_path, 'jobs', job_nr + '.job'), 'rb') as my_file:
    array = pickle.load(my_file)


# load the execution context
with open(os.path.join(tmp_path, 'ExecutionContext.pickle'), 'rb') as my_file:
    ExecutionContext = pickle.load(my_file)


# execute calculation
results_array = []
for element in array:
    try:
        with NamedPrinter(tmp_path, job_nr), \
             ExecutionContext(tmp_path, job_nr):
            single_result = function(element)
    except Exception as e:
        print("execute_sge_array_job: Exception in sge-worker path=",
              tmp_path, 'jobnr=', job_nr, "exception", e, file=sys.stderr)
        single_result = e
    else:
        pass
    finally:
        results_array.append(single_result)

# store result
with open(os.path.join(tmp_path, 'results',
                       job_nr + '.result'), 'wb') as my_file:
    cloudpickle.dump(results_array, my_file)
