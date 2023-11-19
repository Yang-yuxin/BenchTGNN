GPU = '3'
FEAT_DIM = 100 
NUM_FEAT = 100000000  # 37GB feats
NUM_SLICE = 500000  # slice 190MB each time, 1800 * 11 * 25
NUM_RUN = 1000
SKIP = 5

# RESULT: 
# Direct Slice Time:        0.0687ms
# Pinned Buffer Slice Time: 0.0295ms
# UVA Slice Time:           0.0153ms

# Direct Slice Time:        0.0652ms
# Pinned Buffer Slice Time: 0.0241ms
# UVA Slice Time:           0.0155ms

import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

import torch
import dgl
import time

f = torch.randn(NUM_FEAT, FEAT_DIM)
f_pin = f.pin_memory()
pinned_buffer = torch.zeros((NUM_SLICE, FEAT_DIM)).pin_memory()

idxs_cpu = list()
idxs_gpu = list()
for _ in range(NUM_RUN + SKIP):
    idxs_cpu.append(torch.randint(NUM_FEAT, (NUM_SLICE,), device='cpu'))
    idxs_gpu.append(idxs_cpu[-1].cuda())

torch.cuda.synchronize()
for i in range(NUM_RUN + SKIP):
    if i == SKIP:
        t_s = time.time()
    fs = torch.index_select(f, 0, idxs_cpu[i])
    fs = fs.cuda()
torch.cuda.synchronize()
t_avg = (time.time() - t_s) / NUM_RUN
print('Direct Slice Time:        {:.4f}ms'.format(t_avg))

torch.cuda.synchronize()
for i in range(NUM_RUN + SKIP):
    if i == SKIP:
        t_s = time.time()
    torch.index_select(f, 0, idxs_cpu[i], out=pinned_buffer)
    fs = pinned_buffer.cuda()
torch.cuda.synchronize()
t_avg = (time.time() - t_s) / NUM_RUN
print('Pinned Buffer Slice Time: {:.4f}ms'.format(t_avg))

torch.cuda.synchronize()
for i in range(NUM_RUN + SKIP):
    if i == SKIP:
        t_s = time.time()
    fs = dgl.utils.gather_pinned_tensor_rows(f_pin, idxs_gpu[i])
    # data[index] = fs
    # gather(f_pin, slice_idx, out, out_idx)
torch.cuda.synchronize()
t_avg = (time.time() - t_s) / NUM_RUN
print('UVA Slice Time:           {:.4f}ms'.format(t_avg))