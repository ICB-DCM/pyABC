from pyabc.sge import SGE


def test_sge_setup():
    # This test is not sufficient. Currently, there is no use case
    # to use pyabc.sge. Given one, the tests should be extended
    # considerably (which would require installing the gridengine
    # on the test system first).
    sge = SGE(priority=-500, memory="1G", name="test", time_h=1)
    repr(sge)
