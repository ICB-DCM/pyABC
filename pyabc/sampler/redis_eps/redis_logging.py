import os
import logging
import pandas as pd

logger = logging.getLogger("Redis-Sampler")


class RedisSamplerLogger:
    """A logger for the redis sampler with enhanced interest-variables output.
    """

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        if log_file:
            if os.path.exists(log_file) and os.stat(log_file).st_size > 0:
                raise ValueError(
                    f"Sampler log file {log_file} exists and is not empty.")
            if not log_file.endswith('.csv'):
                raise ValueError(
                    f"Sampler log  file {log_file} must be a .csv file")

        # data is a list that can be translated to pandas
        self.data = []

        # internal counters
        self.n_lookahead: int = 0
        self.n_lookahead_accepted: int = 0
        self.n_preliminary: int = 0

    def reset_counters(self):
        """Reset counters. Typically at the start of a generation."""
        self.n_lookahead = 0
        self.n_lookahead_accepted = 0
        self.n_preliminary = 0

    def add_row(self, t: int, n_evaluated: int, n_accepted: int,
                n_lookahead: int = None, n_lookahead_accepted: int = None,
                n_preliminary: int = None):
        """Add row for time `t`."""
        # apply defaults
        if n_lookahead is None:
            n_lookahead = self.n_lookahead
        if n_lookahead_accepted is None:
            n_lookahead_accepted = self.n_lookahead_accepted
        if n_preliminary is None:
            n_preliminary = self.n_preliminary

        # append a row
        self.data.append({
            't': t, 'n_evaluated': n_evaluated, 'n_accepted': n_accepted,
            'n_lookahead': n_lookahead,
            'n_lookahead_accepted': n_lookahead_accepted,
            'n_preliminary': n_preliminary})

    def write(self):
        """Write data to output."""
        # write to screen
        logger.debug(f"Sampling for time t: {self.data[-1]}")
        # write to file
        if self.log_file:
            df = pd.DataFrame(self.data)
            df.to_csv(self.log_file, sep=',')
