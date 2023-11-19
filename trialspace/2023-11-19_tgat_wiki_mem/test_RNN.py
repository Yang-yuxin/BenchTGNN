import argparse
import time
from typing import List

import torch
import yaml
import torch.nn as nn
import tracemalloc

tracemalloc.start()


rnn=nn.GRUCell(100,100).to('cuda')
x=torch.ones((1000,100),device='cuda')
tracemalloc.start(25)
while True:
   print(tracemalloc.get_traced_memory())
   x_bk = x.clone().detach()
   x=rnn(x_bk)
   loss=torch.sum(x)
   loss.backward()