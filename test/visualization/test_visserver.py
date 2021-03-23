"""Test visserver."""

import os
import tempfile
import pytest
import numpy as np

import pyabc
import pyabc.visserver.server as server


db_path = "sqlite:///" + tempfile.mkstemp(suffix='.db')[1]


def setup_module():
    """Run an analysis and create a database.

    Called once at the beginning.
    """
    def model(p):
        return {'ss0': p['p0'] + 0.1 * np.random.uniform(),
                'ss1': p['p1'] + 0.1 * np.random.uniform()}

    p_true = {'p0': 3, 'p1': 4}
    observation = {'ss0': p_true['p0'], 'ss1': p_true['p1']}
    limits = {'p0': (0, 5), 'p1': (1, 8)}
    prior = pyabc.Distribution(**{
        key: pyabc.RV(
            'uniform', limits[key][0], limits[key][1] - limits[key][0])
        for key in p_true.keys()})
    distance = pyabc.PNormDistance(p=2)

    abc = pyabc.ABCSMC(model, prior, distance, population_size=50)
    abc.new(db_path, observation)
    abc.run(minimum_epsilon=.1, max_nr_populations=4)


def teardown_module():
    """Remove history.

    Called once after all tests.
    """
    os.remove(db_path[len("sqlite:///"):])


@pytest.fixture
def client():
    """A fake server client."""
    history = pyabc.History(db_path)
    server.app.config["HISTORY"] = history
    with server.app.test_client() as client:
        yield client


def test_visserver_basic(client):
    """Test that the visserver does at least something."""
    ok_requests = [
        '/',
        '/abc',
        '/abc/1',
        '/abc/1/model/0/t/3',
        '/info',
    ]
    for request in ok_requests:
        rv = client.get(request)
        assert rv.status == '200 OK'

    internal_server_error_requests = [
        '/abc/2',
    ]
    for request in internal_server_error_requests:
        rv = client.get(request)
        assert rv.status == '500 INTERNAL SERVER ERROR'

    not_found_requests = [
        '/britney',
    ]
    for request in not_found_requests:
        rv = client.get(request)
        assert rv.status == '404 NOT FOUND'
