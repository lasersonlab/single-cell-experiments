import pywren
import numpy as np
import sys

def my_function(x):
    return x + 7
print(my_function(3))

def version(x):
    return sys.version_info[0:2]

print(version(None))

wrenexec = pywren.default_executor()
future = wrenexec.call_async(version, 3)
print(future.result())

futures = wrenexec.map(my_function, range(10))
print(pywren.get_all_results(futures))