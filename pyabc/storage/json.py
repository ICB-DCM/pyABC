import json


def save_dict_to_json(dct, log_file):
    """
    Save dict to file.
    """
    with open(log_file, 'w') as f:
        json.dump(dct, f)


def load_dict_from_json(log_file, key_type: type = int):
    """
    Read in json file.
    Convert keys to `key_type'.
    """
    with open(log_file, 'r') as f:
        _dct = json.load(f)
    dct = {}
    for key, val in _dct.items():
        dct[int(key)] = val
    return dct
