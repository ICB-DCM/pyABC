import os
import sqlite3
import time
import redis

from .config import get_config


def within_time(job_start_time, max_run_time_h):
    return time.time() - job_start_time < max_run_time_h * 1.1 * 3600


class SQLiteJobDB:
    SQLITE_DB_TIMEOUT = 2000

    def __init__(self, tmp_dir):
        self.connection = sqlite3.connect(os.path.join(tmp_dir, 'status.db'),
                                          timeout=self.SQLITE_DB_TIMEOUT)

    def clean_up(self):
        pass

    def create(self, nr_jobs):
        # create database for job information
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS "
                "status(ID INTEGER, status TEXT, time REAL)")

    def start(self, ID):
        with self.connection:
            self.connection.execute(
                "INSERT INTO status VALUES(?,?,?)",
                (ID, 'started', time.time()))

    def finish(self, ID):
        with self.connection:
            self.connection.execute(
                "INSERT INTO status VALUES(?,?,?)",
                (ID, 'finished', time.time()))

    def wait_for_job(self, ID, max_run_time_h):
        """
        Return true if we should still wait for the job.
        Return false otherwise
        """
        with self.connection:
            results = self.connection.execute(
                "SELECT status, time from status WHERE ID="
                + str(ID)).fetchall()
            nr_rows = len(results)

        if nr_rows == 0:  # job not jet started
            return True
        if nr_rows == 1:  # job already started
            job_start_time = results[0][1]
            # job took to long
            if not within_time(job_start_time, max_run_time_h):
                print('Job ' + str(ID) + ' timed out.')  # noqa: T001
                return False  # job took too long
            else:  # still time left
                return True
        if nr_rows == 2:  # job finished
            return False
        # something not catched here
        raise Exception('Something went wrong. nr_rows={}'.format(nr_rows))


class RedisJobDB:
    FINISHED_STATE = "finished"
    STARTED_STATE = "started"

    @staticmethod
    def server_online(cls):
        try:
            redis.Redis(cls.HOST).get(None)
        except redis.ConnectionError:
            return False
        else:
            return True

    def __init__(self, tmp_dir):
        config = get_config()
        self.HOST = config["REDIS"]["HOST"]
        self.job_name = os.path.basename(tmp_dir)
        self.connection = redis.Redis(host=self.HOST, decode_responses=True)

    def key(self, ID):
        return self.job_name + ":" + str(ID)

    def clean_up(self):
        IDs = map(int, self.connection.lrange(self.job_name, 0, -1))
        pipeline = self.connection.pipeline()
        for ID in IDs:
            pipeline.delete(self.key(ID))
        pipeline.delete(self.job_name)
        pipeline.execute()

    def create(self, nr_jobs):
        pipeline = self.connection.pipeline()
        for ID in range(1, nr_jobs+1):
            pipeline.rpush(self.job_name, ID)
        pipeline.execute()

    def start(self, ID):
        self.connection.hmset(self.key(ID), {"status": self.STARTED_STATE,
                                             "time": time.time()})

    def finish(self, ID):
        self.connection.hmset(self.key(ID), {"status": self.FINISHED_STATE,
                                             "time": time.time()})

    def wait_for_job(self, ID, max_run_time_h):
        values = self.connection.hgetall(self.key(ID))
        if len(values) == 0:  # not yet set, job not yet started
            return True

        status = values["status"]
        time_stamp = float(values["time"])

        if status == self.FINISHED_STATE:
            return False

        if status == self.STARTED_STATE:
            if within_time(time_stamp, max_run_time_h):
                return True
            return False

        raise Exception('Something went wrong.')


def job_db_factory(tmp_path):
    """

    Returns
    -------
    SQLite or redis db depending on availability
    """
    config = get_config()
    if config["BROKER"]["TYPE"] == "REDIS":
        return RedisJobDB(tmp_path)
    if config["BROKER"]["TYPE"] == "SQLITE":
        return SQLiteJobDB(tmp_path)
    raise Exception("Unknown broker: {}".format(config["BROKER"]["TYPE"]))
