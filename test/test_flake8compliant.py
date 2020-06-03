import subprocess
import sys
import pathlib


def test_flake8compliant():
    path = pathlib.Path(__file__).parent.parent
    exec_ = sys.executable
    cmd = f"cd {path}; {exec_} -m flake8 pyabc test " \
        "--per-file-ignores='*/__init__.py:F401'"
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    print(r.stdout.decode())
    print(r.stderr.decode())
    assert r.stdout.decode() == ""
    assert r.stderr.decode() == ""
