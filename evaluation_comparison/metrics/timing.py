from collections import namedtuple
import json
import pickle
import time

import numpy as np

__all__ = [
    "enable_timing", "disable_timing",
    "time_lap", "time_last", "time_accum",
    "get_time_tables",
    "pickle_time_table", "json_time_table"
]

# Dictionaries for storing the global time durations
_TIME_LAP = {}
_TIME_LAST = {}
_TIME_ACCUM = {}


# Global variable for enabling or disabling timing functions
_ENABLE_TIME = True


# Functions for aggregating durations on the respective dictionaries
def _agg_append(*, value, location):
    if location is None:
        return [value]

    location.append(value)
    return location


def _agg_last(*, value, location):
    if location is None:
        return value

    location = value
    return location


def _agg_accum(*, value, location):
    if location is None:
        return value

    location += value
    return location


# Functions for aggregating statistics
def _agg_stats_func(*, values):
    np_values = np.array(values)

    count = int(len(np_values))
    val_sum = float(np_values.sum())
    mean = float(val_sum / count)
    median = float(np.median(np_values))
    val_max = float(np.max(np_values))
    val_min = float(np.min(np_values))
    std_dev = float(np_values.std())

    stats = {
        "count": count,
        "sum": val_sum,
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "max": val_max,
        "min": val_min
    }
    return stats


def _agg_stats_lap(*, stats_dict):
    return {label: {func: _agg_stats_func(values=values) for func, values in subdict.items()}
            for label, subdict in stats_dict.items()}


_TimeDictCfg = namedtuple("TimeDictCfg", ["dict", "agg_func", "stats_func"])

_DICT_AGG_FUNCTIONS = {
    "lap":   _TimeDictCfg(dict=_TIME_LAP,   agg_func=_agg_append, stats_func=_agg_stats_lap),
    "last":  _TimeDictCfg(dict=_TIME_LAST,  agg_func=_agg_last,   stats_func=None),
    "accum": _TimeDictCfg(dict=_TIME_ACCUM, agg_func=_agg_accum,  stats_func=None),
}


def _insert_val(*, dict_agg_key, label, func_name, value):

    dict_agg_cfg = _DICT_AGG_FUNCTIONS[dict_agg_key]
    time_dict = dict_agg_cfg.dict
    agg_func = dict_agg_cfg.agg_func

    if label not in time_dict:
        time_dict[label] = {func_name: agg_func(value=value, location=None)}
        return

    if func_name not in time_dict[label]:
        time_dict[label][func_name] = agg_func(value=value, location=None)
        return

    time_dict[label][func_name] = agg_func(value=value, location=time_dict[label][func_name])


def _time_and_insert(*, func, args, kwargs, dict_agg_key, label):
    start_time = time.time()
    func_name = func.__name__
    result = func(*args, **kwargs)
    duration = time.time() - start_time

    _insert_val(dict_agg_key=dict_agg_key, label=label, func_name=func_name, value=duration)

    return result


def _conditional_time_and_insert(*, func, args, kwargs, dict_agg_key, label):
    global _ENABLE_TIME

    if _ENABLE_TIME:
        return _time_and_insert(func=func, args=args, kwargs=kwargs, dict_agg_key=dict_agg_key, label=label)

    return func(*args, **kwargs)


def set_timing_enabled(enabled: bool):
    global _ENABLE_TIME
    _ENABLE_TIME = enabled


def disable_timing():
    set_timing_enabled(False)


def enable_timing():
    set_timing_enabled(True)


def toggle_timing():
    global _ENABLE_TIME
    _ENABLE_TIME = not _ENABLE_TIME


def is_timing_enabled():
    global _ENABLE_TIME
    return _ENABLE_TIME


def time_lap(label):
    def decorator(func):
        def wrap(*args, **kwargs):
            return _conditional_time_and_insert(func=func, args=args, kwargs=kwargs, dict_agg_key="lap", label=label)

        return wrap
    return decorator


def time_last(label):
    def decorator(func):
        def wrap(*args, **kwargs):
            return _conditional_time_and_insert(func=func, args=args, kwargs=kwargs, dict_agg_key="last", label=label)

        return wrap
    return decorator


def time_accum(label):
    def decorator(func):
        def wrap(*args, **kwargs):
            return _conditional_time_and_insert(func=func, args=args, kwargs=kwargs, dict_agg_key="accum", label=label)

        return wrap
    return decorator


def get_time_tables(*, get_raw=True, get_agg=True):

    time_tables = {}
    for k, v in _DICT_AGG_FUNCTIONS.items():
        if v.dict:
            if get_raw or v.stats_func is None or not get_agg:
                time_tables[k] = v.dict

            if v.stats_func is not None and get_agg:
                time_tables[f"{k}_stats"] = v.stats_func(stats_dict=v.dict)

    return time_tables


def _dump_time_table(path, *, save_func, file_mode, get_raw=True, get_agg=True):
    if path is None:
        return

    with open(path, file_mode) as f:
        save_func(get_time_tables(get_raw=get_raw, get_agg=get_agg), f)


def _json_dump(obj, fh):
    json.dump(obj, fh, indent=4)


def pickle_time_table(path, *, get_raw=True, get_agg=True):
    _dump_time_table(path, save_func=pickle.dump, file_mode="wb", get_raw=get_raw, get_agg=get_agg)


def json_time_table(path, *, get_raw=True, get_agg=True):
    _dump_time_table(path, save_func=_json_dump, file_mode="w", get_raw=get_raw, get_agg=get_agg)


if __name__ == "__main__":

    @time_lap("addition")
    def sum_values(values):
        return np.sum(values)

    @time_accum("accum_addition")
    def sum_values_accum(values):
        return np.sum(values)

    @time_last("last_addition")
    def sum_values_last(values):
        return np.sum(values)

    disable_timing()

    for i in range(100):
        test_values = np.random.random(100000)

        enable_timing()
        summed_val = sum_values(test_values)
        # disable_timing()
        summed_acc_val = sum_values_accum(test_values)

        print(i, sum_values(test_values), sum_values_accum(test_values), sum_values_last(test_values))

    time_stats = get_time_tables()
    time_stats_wor = get_time_tables(get_raw=False)
    time_stats_woa = get_time_tables(get_agg=False)

    json_time_table("/home/arceyd/Desktop/time_test.json")
    pickle_time_table("/home/arceyd/Desktop/time_test.pickle")
