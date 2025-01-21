import uproot
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
import os,sys
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorchtools import EarlyStopping

from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler, QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")