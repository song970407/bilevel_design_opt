import random

import numpy as np
import torch


def fix_seed(seed_num=0):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
