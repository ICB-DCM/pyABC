import subprocess
import os
import signal
import time


def test_server_dash():
    cmd = "abc-server-dash"
    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE,
                           shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    os.killpg(os.getpgid(proc.pid),
               signal.SIGTERM)  # Send the signal to all the process groups
