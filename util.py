import torch
import random
import numpy as np

def deterministic(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_res(model, x, y, pi=None):
    datapoint = []
    return datapoint