import numpy as np


def nested_dict_with_arrays_cmp(a, b):
    if isinstance(a, np.ndarray):
        return np.allclose(a, b)
    elif isinstance(a, list) and isinstance(a[0], np.ndarray):
        return np.allclose(np.array(a), np.array(b))
    elif not isinstance(a, dict):
        return a == b

    a_keys = set(a.keys())
    b_keys = set(b.keys())
    if len(a_keys ^ b_keys) > 0:
        return False
    else:
        return all((nested_dict_with_arrays_cmp(va, b[k]) for k, va in a.items()))
