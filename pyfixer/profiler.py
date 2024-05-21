#!/usr/bin/env python
# coding: utf-8

import time
import psutil
from memory_profiler import memory_usage

class Profiler:
    def __init__(self):
        pass

    @staticmethod 
    def profile(func, *args, **kwargs):
        # Get the name of the function
        func_name = func.__name__

        # Measure start time and memory usage
        start_time = time.time()
        mem_usage_before = memory_usage()  # Returns the memory usage as a list

        # Measure CPU usage before execution
        cpu_before = psutil.cpu_percent(interval=None)

        # Execute the function
        result = func(*args, **kwargs)

        # Measure end time, memory, and CPU usage
        end_time = time.time()
        mem_usage_after = memory_usage()  # Returns the memory usage as a list
        cpu_after = psutil.cpu_percent(interval=None)

        # Calculate differences and total time
        elapsed_time = end_time - start_time
        mem_used = max(mem_usage_after) - min(mem_usage_before)  # Max memory used during the function execution
        cpu_used = cpu_after - cpu_before

        # Output the performance metrics
        print(f"Function '{func_name}' execution time: {elapsed_time:.3f} seconds")
        print(f"Function '{func_name}' memory used: {mem_used:.3f} MB")
        print(f"Function '{func_name}' CPU usage change: {cpu_used}%")
        print()
        print('-----------------')

        return result


