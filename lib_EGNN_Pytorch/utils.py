# coding=utf-8

# Lint as: python3
"""Helper functions for graph processing."""
import resource
import pickle
import shutil
import os
import csv
import numpy as np
import scipy.sparse


def get_max_memory_bytes():
    return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def save_dict_into_table(data_dict, file_path, file_name):
    """
    Save the dict data into a .txt table
    Args:
        data_dict : a dictionary of any purpose
    """
    target_file = os.path.join(file_path, file_name)
    with open(target_file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        header = ["dict key", "value"]
        wr.writerow(header)
        for key, val in data_dict.items():
            tmp_line = [key, val]
            wr.writerow(tmp_line)

def save_info_pickle(data_path, target_data, exist_ok = False):
    
    if not exist_ok and os.path.exists(data_path):
        print(f"{data_path}, path folder already exists, so removed ...")
        os.remove(data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    with open(data_path, "wb") as fp:
        pickle.dump(target_data, fp)


