import time
import numpy as np

def eta(list_of_times, remaining_batches):
    mean = np.mean(list_of_times)
    remaining = mean * remaining_batches
    return time.strftime("%H:%M:%S", time.gmtime(remaining))
