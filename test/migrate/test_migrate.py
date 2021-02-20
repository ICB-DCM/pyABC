"""Migration tests."""

import pytest
import os
import tempfile
import pyabc


def test_db_import(script_runner):
    """Import an outdated database, assert import raises, and then convert."""
    db_file = os.path.join(tempfile.gettempdir(), "pyabc_test_migrate.db")

    # this database was created with a previos pyabc version, thus
    #  import should fail
    with pytest.raises(AssertionError):
        pyabc.History("sqlite:///" + db_file)

    # call the migration script
    ret = script_runner.run('abc-migrate', '--src', db_file, '--dst', db_file)
    assert ret.success

    # now it should work
    h = pyabc.History("sqlite:///" + db_file)
    h.get_weighted_sum_stats()

    # remove file
    os.remove(db_file)
