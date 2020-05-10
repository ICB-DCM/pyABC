import subprocess
import sys
import pathlib


def test_flake8compliant():
    path = pathlib.Path(__file__).parent.parent
    exec_ = sys.executable
    cmd = "cd {path}; {exec_} -m flake8 pyabc test".format(
        path=path, exec_=exec_)
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    print(r.stdout.decode())
    print(r.stderr.decode())
    assert r.stdout.decode() == ""
    assert r.stderr.decode() == ""
