import subprocess  # noqa: S404


def test_flake8compliant():
    r = subprocess.run(  # noqa: S603
        "./run_flake8.sh",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    print(r.stdout.decode())
    print(r.stderr.decode())
    assert r.stdout.decode() == ""
    assert r.stderr.decode() == ""
