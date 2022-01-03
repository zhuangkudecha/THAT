import pickle

import numpy as np
import torch
from torch import nn

'''
how to copy the structure from the csi to this sensor signal 
'''
#first of all prepare datasets (cross subject/ new user )
#use the pamap2 datasets subject 1-7 as training data  others validate data
datasets_dir = 'THAT/datasets/pamap2/ALL_AccGyroMag_w96.pkl'
with open(datasets_dir, 'rb') as f:
    data = pickle.load(f)
data_x = data[0]
data_label = data[1]
data_subject = data[2]

"""
use pytorch accomplish a attention module 
"""
#test