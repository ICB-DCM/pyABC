import subprocess
import sys
import pathlib


def test_flake8compliant():
    path = pathlib.Path(__file__).parent.parent
    exec_ = sys.executable
    exclude = (path / "flake8_exclude.txt").read_text()
    cmd = "cd {path}; {exec_} -m flake8 --exclude={exclude}".format(
        path=path, exec_=exec_, exclude=exclude)
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    print(r.stdout.decode())
    print(r.stderr.decode())
    assert r.stdout.decode() == ""
    assert r.stderr.decode() == ""
