# coding=utf-8

# Lint as: python3
"""Helper functions for graph processing."""
import resource
import pickle
import shutil
import os
import csv
import numpy as np
import shutil
import dill
import os

import csv
from torch.autograd import Variable



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


### ================== utils from meta learn =======================


def save_info_use_dill(data_path, target_data, exist_ok = False):
    
    if not exist_ok and os.path.exists(os.path.dirname(data_path)):
        print(f"{data_path}, path folder already exists, so removed ...")
        shutil.rmtree(os.path.dirname(data_path))
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    with open(data_path, "wb") as fp:
        dill.dump(target_data, fp)


def to_numpy(x):
    """
        The original purpose of Variables was to be able to use automatic differentiation
        Autograd automatically supports Tensors with requires_grad set to True
    """
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()

##################
# PRINTING UTILS #
#----------------#

_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg, style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1 = _bcolors[style], msg = msg, color2='\033[0m'))



