import pytest
import os
import tempfile
import pyabc


def test_db_import(script_runner):

    db_file = os.path.join(tempfile.gettempdir(), "pyabc_test_migrate.db")
    with pytest.raises(AssertionError):
        pyabc.History("sqlite:///" + db_file)

    ret = script_runner.run('abc-migrate', '--src', db_file, '--dst', db_file)
    assert ret.success

    # now it should work
    h = pyabc.History("sqlite:///" + db_file)
    h.get_weighted_sum_stats()

    os.remove(db_file)
