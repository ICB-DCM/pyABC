"""Migration tests."""

import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pytest

import pyabc
from pyabc.parameters import Parameter
from pyabc.population import Particle, Population
from pyabc.storage.version import __db_version__

# model names of the test database created below
MODEL_NAMES = ['m0', 'm1']

# global particle weights per population index and model, i.e. as stored by
#  the current format: they sum to 1 across all particles of all models,
#  and within a model to that model's probability
WEIGHTS = {
    0: {0: [0.3, 0.1], 1: [0.2, 0.4]},
    1: {0: [0.1, 0.15], 1: [0.5, 0.25]},
}

# wall times per population index
WALL_TIMES = {0: 12.5, 1: 7.25}

# SQLite can only drop table columns from version 3.35 on
requires_drop_column = pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35),
    reason='Dropping a table column requires SQLite>=3.35',
)


def test_db_import(script_runner):
    """Import an outdated database, assert import raises, and then convert."""
    db_file = os.path.join(tempfile.gettempdir(), 'pyabc_test_migrate.db')

    # this database was created with a previos pyabc version, thus
    #  import should fail
    with pytest.raises(AssertionError):
        pyabc.History('sqlite:///' + db_file)

    # call the migration script
    ret = script_runner.run(
        ['abc-migrate', '--src', db_file, '--dst', db_file]
    )
    assert ret.success

    # now it should work
    h = pyabc.History('sqlite:///' + db_file)
    h.get_weighted_sum_stats()

    # remove file
    os.remove(db_file)


def create_current_db(db_file: str) -> None:
    """Create a database in the current format, holding two models."""
    h = pyabc.History('sqlite:///' + db_file)
    h.store_initial_data(None, {}, {'ss': 0.0}, {}, MODEL_NAMES, '', '', '{}')
    for t, weights in WEIGHTS.items():
        particles = [
            Particle(
                m=m,
                parameter=Parameter({'a': float(m), 'b': float(ix)}),
                weight=w,
                sum_stat={'ss': float(ix)},
                distance=0.1 * (ix + 1),
            )
            for m, ws in weights.items()
            for ix, w in enumerate(ws)
        ]
        h.append_population(
            t,
            0.5 / (t + 1),
            Population(particles),
            10,
            MODEL_NAMES,
            wall_time=WALL_TIMES[t],
        )


def query(db_file: str, sql: str) -> list:
    """Run a raw SQL query on the database file."""
    con = sqlite3.connect(db_file)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def db_version(db_file: str) -> str:
    """Storage format version of the database."""
    return str(query(db_file, 'SELECT version_num FROM version')[0][0])


def columns(db_file: str, table: str) -> list[str]:
    """Column names of a database table."""
    return [row[1] for row in query(db_file, f'PRAGMA table_info({table})')]


def stored_weights(db_file: str) -> dict:
    """Stored particle weights as ``{t: {m: sorted weights}}``.

    The pre-population is not included.
    """
    rows = query(
        db_file,
        'SELECT populations.t, models.m, particles.w FROM particles '
        'JOIN models ON models.id = particles.model_id '
        'JOIN populations ON populations.id = models.population_id '
        'WHERE populations.t >= 0',
    )
    weights = {}
    for t, m, w in rows:
        weights.setdefault(t, {}).setdefault(m, []).append(w)
    return {
        t: {m: sorted(ws) for m, ws in per_model.items()}
        for t, per_model in weights.items()
    }


def global_weights() -> dict:
    """Expected weights in the current format, summing to 1 over all models."""
    return {
        t: {m: sorted(ws) for m, ws in per_model.items()}
        for t, per_model in WEIGHTS.items()
    }


def per_model_weights() -> dict:
    """Expected weights in version 1, summing to 1 within each model."""
    return {
        t: {m: sorted(np.asarray(ws) / sum(ws)) for m, ws in per_model.items()}
        for t, per_model in WEIGHTS.items()
    }


def assert_weights_close(actual: dict, expected: dict) -> None:
    """Assert that two ``{t: {m: weights}}`` dictionaries match."""
    assert actual.keys() == expected.keys()
    for t, per_model in expected.items():
        assert actual[t].keys() == per_model.keys()
        for m, ws in per_model.items():
            assert np.allclose(actual[t][m], ws)


def alembic_config(db: str):
    """Alembic configuration, skipping the test if alembic is missing."""
    pytest.importorskip('alembic')
    from pyabc.storage.migrate import _alembic_config

    return _alembic_config(db)


def to_v1(db_file: str) -> None:
    """Turn a current-format database into a version 1 database.

    Applies the version 2 downgrade, which reverts the weight normalization
    (from global back to within-model, ``w = g_i / p_model``), and in addition
    drops the wall time column, which did not exist in version 1 but is kept
    by the downgrade.
    """
    command = pytest.importorskip('alembic.command')
    command.downgrade(alembic_config(db_file), '1')

    con = sqlite3.connect(db_file)
    with con:
        con.execute('ALTER TABLE populations DROP COLUMN wall_time')
    con.close()


@requires_drop_column
def test_migrate_v1_to_v2(script_runner, tmp_path):
    """Migrating a version 1 database to the current format.

    Checks that the wall time column is added and that particle weights are
    converted from the per-model to the global normalization, on a database
    with two models, i.e. with model probabilities != 1.
    """
    src = str(tmp_path / 'v1.db')
    dst = str(tmp_path / 'v2.db')

    # create a database in the current format and record reference values
    create_current_db(src)
    h = pyabc.History('sqlite:///' + src)
    p_models = {t: h.get_model_probabilities(t=t) for t in WEIGHTS}
    distributions = {
        (t, m): h.get_distribution(m=m, t=t) for t in WEIGHTS for m in [0, 1]
    }
    assert_weights_close(stored_weights(src), global_weights())

    # turn it into a version 1 database
    to_v1(src)
    assert db_version(src) == '1'
    assert 'wall_time' not in columns(src, 'populations')
    assert_weights_close(stored_weights(src), per_model_weights())

    # an outdated database cannot be imported
    with pytest.raises(AssertionError, match='Database has version 1'):
        pyabc.History('sqlite:///' + src)

    # call the migration script
    ret = script_runner.run(['abc-migrate', '--src', src, '--dst', dst])
    assert ret.success

    # the source database is left untouched
    assert db_version(src) == '1'
    assert 'wall_time' not in columns(src, 'populations')
    assert_weights_close(stored_weights(src), per_model_weights())

    # the destination database is up-to-date and has the new column
    assert db_version(dst) == __db_version__ == '2'
    assert 'wall_time' in columns(dst, 'populations')

    # weights are back to the global normalization
    assert_weights_close(stored_weights(dst), global_weights())

    # the pre-population's dummy particle is not affected
    assert query(
        dst,
        'SELECT particles.w FROM particles '
        'JOIN models ON models.id = particles.model_id '
        'JOIN populations ON populations.id = models.population_id '
        'WHERE populations.t = -1',
    ) == [(1.0,)]

    # the migrated database gives the same results as the original one
    h = pyabc.History('sqlite:///' + dst)
    for t in WEIGHTS:
        assert np.allclose(
            h.get_model_probabilities(t=t).p.values, p_models[t].p.values
        )
        for m in [0, 1]:
            df, w = h.get_distribution(m=m, t=t)
            df_expected, w_expected = distributions[(t, m)]
            pd.testing.assert_frame_equal(df, df_expected)
            # within-model weights sum to 1
            assert np.allclose(w, w_expected)
            assert np.isclose(w.sum(), 1.0)

    # wall times are unknown for migrated populations
    populations = h.get_all_populations()
    assert 'wall_time' in populations.columns
    assert populations.wall_time.isna().all()


def test_migrate_v2_downgrade(tmp_path):
    """The version 2 revision can be reverted and then applied again.

    The downgrade inverts the weight conversion, but keeps the wall time
    column, so that the upgrade must tolerate an existing column.
    """
    command = pytest.importorskip('alembic.command')

    db_file = str(tmp_path / 'db.db')
    create_current_db(db_file)
    cfg = alembic_config(db_file)

    # revert to version 1
    command.downgrade(cfg, '1')
    assert db_version(db_file) == '1'
    assert 'wall_time' in columns(db_file, 'populations')
    assert_weights_close(stored_weights(db_file), per_model_weights())

    # and migrate back to the current version
    command.upgrade(cfg, 'head')
    assert db_version(db_file) == __db_version__
    assert 'wall_time' in columns(db_file, 'populations')
    assert_weights_close(stored_weights(db_file), global_weights())


def test_db_identifier(tmp_path):
    """Databases can be specified as file names or as sqlite URLs."""
    pytest.importorskip('alembic')
    from pyabc.storage.migrate import _alembic_config, _to_db_file

    db_file = str(tmp_path / 'db.db')

    # file names and URLs are equivalent
    assert _to_db_file(db_file) == _to_db_file('sqlite:///' + db_file)
    assert _alembic_config(db_file).get_main_option(
        'sqlalchemy.url'
    ) == _alembic_config('sqlite:///' + db_file).get_main_option(
        'sqlalchemy.url'
    )

    # other dialects are not supported
    with pytest.raises(ValueError, match='only supports sqlite'):
        _to_db_file('postgresql://user@localhost/db')


def test_migrate_unsupported_dialect(script_runner, tmp_path):
    """Migrating a non-sqlite database gives an error message."""
    ret = script_runner.run(
        [
            'abc-migrate',
            '--src',
            'postgresql://user@localhost/db',
            '--dst',
            str(tmp_path / 'db.db'),
        ]
    )
    assert 'only supports sqlite' in ret.stdout
