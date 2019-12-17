import json


def save_pnorms(dct, log_file):
    with open(log_file, 'w') as f:
        json.dump(dct, f)


def load_pnorms(log_file):
    with open(log_file, 'r') as f:
        _dct = json.load(f)
    dct = {}
    for key, val in _dct.items():
        dct[int(key)] = float(val)
    return dct
