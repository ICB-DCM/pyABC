import sys

from pyabc.sge.db import job_db_factory

tmp_path = sys.argv[1]
job_nr = sys.argv[2]

# save end time to database
job_db = job_db_factory(tmp_path)
job_db.finish(int(job_nr))
