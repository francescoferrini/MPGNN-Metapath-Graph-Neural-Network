
from errno import EDEADLK
import argparse
from resource import struct_rusage
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import math
import numpy as np
import torch
from torch import Tensor


labels = pd.read_pickle('/Users/francescoferrini/VScode/metapathCompetitors/Graph_Transformer_Networks/data/DBLP/labels.pkl')
print(len(labels[2]))



