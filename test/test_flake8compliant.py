import subprocess
import sys
import pathlib


def test_flake8compliant():
    path = pathlib.Path(__file__)
    cmd = ("cd {path}; {exec} -m flake8 --exclude={exclude}"
           .format(exec=sys.executable,
                   exclude=(
                       path.parent.parent / "flake8_exclude.txt").read_text(),
                   path=path.parent.parent))
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    print(r.stdout.decode())
    print(r.stderr.decode())
    assert r.stdout.decode() == ""
    assert r.stderr.decode() == ""
