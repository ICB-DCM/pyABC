import cProfile as profile
import os


class ExecutionContextMixin:
    keep_output_directory = False

    def __init__(self, tmp_path, job_nr):
        self.tmp_path = tmp_path
        self.job_nr = job_nr


class NamedPrinter(ExecutionContextMixin):
    """
    Context with appends the job name and number to anything
    printed by that job.
    """
    def __init__(self, tmp_path, job_nr):
        super().__init__(tmp_path, job_nr)
        import sys
        self.original_stdout_write = sys.stdout.write
        self.original_stderr_write = sys.stderr.write
        self.name_tag = "[{}:{}]".format(os.path.basename(self.tmp_path),
                                         self.job_nr)

    def __enter__(self):
        import sys
        sys.stdout.write = self.named_write_stdout
        sys.stderr.write = self.named_write_stderr

    def __exit__(self, exc_type, exc_value, traceback):
        import sys
        sys.stdout.write = self.original_stdout_write
        sys.stderr.write = self.original_stderr_write

    def process_text(self, text):
        return text.replace("\n",  self.name_tag + "\n")

    def named_write_stdout(self, text):
        self.original_stdout_write(self.process_text(text))

    def named_write_stderr(self, text):
        self.original_stderr_write(self.process_text(text))


class DefaultContext(ExecutionContextMixin):
    """
    Does nothing special.
    """
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ProfilingContext(ExecutionContextMixin):
    """
    Profiles the running jobs and stores the profiles in the
    temporary job folder
    in the subdirectory "profiling".

    Useful for debugging. Do not use in production.
    """
    RELATIVE_OUTPUT_FOLDER = "profiling"
    keep_output_directory = True

    def __init__(self, tmp_path, job_nr):
        super().__init__(tmp_path, job_nr)
        self._profile = profile.Profile()

    def __enter__(self):
        self._profile.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._profile.disable()
        self._dump_profile()

    def _dump_profile(self):
        output_directory = os.path.join(self.tmp_path,
                                        self.RELATIVE_OUTPUT_FOLDER)
        self._make_output_directory(output_directory)
        output_file = os.path.join(output_directory,
                                   str(self.job_nr) + ".pstats")
        self._profile.dump_stats(output_file)

    @staticmethod
    def _make_output_directory(output_directory):
        try:
            os.makedirs(output_directory)
        except FileExistsError:
            pass
