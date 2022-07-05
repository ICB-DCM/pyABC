import multiprocessing
import time

from pyabc.visserver import server_dash


def call_dash_server(ret_value):
    try:
        server_dash.run_app()
    except Exception as e:
        ret_value.value = 0
        print(e)


def test_dash_server():
    ret_value = multiprocessing.Value("i", 1, lock=False)
    t1 = multiprocessing.Process(target=call_dash_server, args=[ret_value])
    t1.start()
    time.sleep(3)
    t1.terminate()
    assert ret_value.value == 1
