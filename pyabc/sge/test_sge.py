from time import sleep

from pyabc.sge import SGE

if __name__ == "__main__":

    def f(x):
        sleep(30)
        return x * 2

    print("Start sge test", flush=True)  # noqa: T001
    sge = SGE(priority=0, memory="1G", name="test", time_h=1)

    print("Do map", flush=True)  # noqa: T001
    res = sge.map(f, [1, 2, 3, 4])

    print("Got results", flush=True)  # noqa: T001
    if res != [2, 4, 6, 8]:
        raise AssertionError("Wrong result, got {}".format(res))

    print("Finished", flush=True)  # noqa: T001
