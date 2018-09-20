import pywren
import time

import datetime
from pywren.wait import wait, ALL_COMPLETED

N = 130

def sleep_test(x):
    time.sleep(x)
    return x + 1

def get_all_results(fs):
    """
    Take in a list of futures and block until they are repeated,
    call result on each one individually, and return those
    results.

    Will throw an exception if any future threw an exception
    """
    print("waiting", datetime.datetime.now())
    wait(fs, return_when=ALL_COMPLETED)
    print("iterate through results", datetime.datetime.now())
    return [f.result() for f in fs]

print("starting", datetime.datetime.now())
wrenexec = pywren.default_executor()
futures = wrenexec.map(sleep_test, [10] * N)
print("finished map", datetime.datetime.now())
get_all_results(futures)
print("exiting", datetime.datetime.now())
