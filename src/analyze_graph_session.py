import argparse
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import random
from scipy.optimize import minimize
from scipy.stats import norm

BINS = 100

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--high_freq', default=False, action='store_true')
parser.add_argument('--add_reverse', default=False, action='store_true')

args = parser.parse_args()

print(args)

df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
s = 'src' if 'src' in df.columns else 'u'
d = 'dst' if 'dst' in df.columns else 'i'
t = 'time' if 'time' in df.columns else 'ts'
num_nodes = max(int(df[s].max()), int(df[d].max())) + 1
print('num_nodes: ', num_nodes)
# import pdb; pdb.set_trace()
ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row[s])
    dst = int(row[d])
    
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row[t])
    ext_full_eid[src].append(idx)
    
    if args.add_reverse:
        ext_full_indices[dst].append(src)
        ext_full_ts[dst].append(row[t])
        ext_full_eid[dst].append(idx)

# if args.data == 'WIKI':
#     for i in range(len(ext_full_ts)):
#         random.shuffle(ext_full_ts[i])

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

def get_inter_activity_time(src):
    neighs, ts, eid  = ext_full_indices[src], ext_full_ts[src], ext_full_eid[src]
    intervals = []
    i = 0
    while i < len(neighs):
        ni = neighs[i]
        j = i
        while j < len(neighs)-1:
            nj = neighs[j]
            nk = neighs[j+1]
            if (ni == nj and ni != nk):
                intervals.append(ts[j] - ts[i])
                break
            elif j == len(neighs) - 2:
                intervals.append(ts[j+1] - ts[i])
                break
            else:
                j += 1
            
        i = j + 1
    return intervals

def get_activity_density(src):
    neighs, ts, eid  = ext_full_indices[src], ext_full_ts[src], ext_full_eid[src]
    intervals = []
    i = 0
    while i < len(neighs):
        ni = neighs[i]
        j = i
        l = 0
        while j < len(neighs)-1:
            nj = neighs[j]
            nk = neighs[j+1]
            if (ni == nj and ni != nk):
                intervals.append(j - i)
                break
            elif j == len(neighs) - 2:
                intervals.append(j + 1 - i)
                break
            else:
                j += 1
        i = j + 1
    return intervals
    
def tsort_original(i, indices, ts, eid):
    if not len(indices):
        return
    try:
        sidx = np.argsort(ts)
        indices = np.array(indices)[sidx]
        ts = np.array(ts)[sidx]
        eid = np.array(eid)[sidx]
    except TypeError:
        import pdb; pdb.set_trace()

print('sorting and analyzing...')
t_inter_act = []
for i in tqdm(range(num_nodes)):
    tsort_original(i, ext_full_indices[i], ext_full_ts[i], ext_full_eid[i])
    t_inter_act += get_inter_activity_time(i)

t_inter_act = np.array(t_inter_act)
p1 = np.percentile(t_inter_act, 0.01)
p99 = np.percentile(t_inter_act, 99.99)

# Filtering the array to remove the smallest and largest 1%
filtered_data = t_inter_act[(t_inter_act > p1) & (t_inter_act < p99)]
# filtered_data = t_inter_act
x = pd.Series(filtered_data)
# x = pd.Series(t_inter_act)
# import pdb; pdb.set_trace()
# histogram on linear scale
plt.subplot(211)
x_log = np.log10(x)
hist, bins = np.histogram(x_log, BINS, density=True)
_ = plt.hist(x, BINS, density=True, stacked=True)
if args.data == 'Flights':
    xlabel = f'time (day)'
else:
    xlabel = f'time (second)'
plt.xlabel(xlabel)
plt.subplot(212)

# Bimodal distribution function
def bimodal(x, mu1, sigma1, mu2, sigma2, weight1):
    return weight1*norm.pdf(x, mu1, sigma1) + (1-weight1)*norm.pdf(x, mu2, sigma2)

# Constraint function
def tri_alpha_constraint(params):
    # Assuming params[0] = alpha1, params[1] = alpha2
    alpha1, alpha2 = params[-2], params[-1]
    return 1 - (alpha1 + alpha2)  # This needs to be >= 0

# Define the constraint dictionary
cons = ({'type': 'ineq', 'fun': tri_alpha_constraint})

# Bimodal distribution function
def trimodal(x, mu1, sigma1, mu2, sigma2, mu3, sigma3, weight1, weight2 ):
    return weight1*norm.pdf(x, mu1, sigma1) + weight2*norm.pdf(x, mu2, sigma2) + (1- weight1-weight2) *norm.pdf(x, mu3, sigma3)

# Objective function to minimize (Sum of Squared Differences)
def objective(params, bin_edges, counts):
    mu1, sigma1, mu2, sigma2, weight1 = params
    # Calculate the predicted counts for each bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    predicted_counts = bimodal(bin_centers, mu1, sigma1, mu2, sigma2, weight1) * sum(counts) * (bin_edges[1] - bin_edges[0])
    # Calculate SSD
    ssd = np.sum((counts - predicted_counts) ** 2)
    return ssd

# Objective function to minimize (Sum of Squared Differences)
def objective3(params, bin_edges, counts):
    mu1, sigma1, mu2, sigma2, mu3, sigma3, weight1, weight2 = params
    # Calculate the predicted counts for each bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    predicted_counts = trimodal(bin_centers, mu1, sigma1, mu2, sigma2, mu3, sigma3, weight1, weight2) * sum(counts) * (bin_edges[1] - bin_edges[0])
    # Calculate SSD
    ssd = np.sum((counts - predicted_counts) ** 2)
    return ssd

# Initial guesses for parameters
# Perform the optimization
if not args.high_freq:
    initial_params = [2, 0.1, 5, 0.1, 0.5]
    result = minimize(objective, initial_params, args=(bins, hist), bounds=[(0, 10), (0.01, 1e3), (0, 10), (0.01, 1e3), (0.2, 1)])
    mu1, sigma1, mu2, sigma2, weight1 = result.x
    normal1_pdf = norm.pdf(bins, mu1, sigma1) * weight1
    normal2_pdf = norm.pdf(bins, mu2, sigma2) * (1 - weight1)
    bimodal_pdf = bimodal(bins, mu1, sigma1, mu2, sigma2, weight1)
    plt.plot(bins, bimodal_pdf, label='Fitted bimodal distribution', color='red', linewidth=1)
    plt.plot(bins, normal1_pdf, label='Normal distribution 1', color='green', linestyle='--', linewidth=1)
    plt.plot(bins, normal2_pdf, label='Normal distribution 2', color='blue', linestyle='--', linewidth=1)
else:
    initial_params = [0, 0.1, 1, 0.1, 5, 0.1, 0.33, 0.33]
    result = minimize(objective3, initial_params, args=(bins, hist), 
                      bounds=[(0, 10), (0.01, 1e3),(0, 10), (0.01, 1e3), (0, 10), (0.01, 1e3), (0.2, 1), (0.2, 1)],
                      constraints=cons)
    mu1, sigma1, mu2, sigma2, mu3, sigma3, weight1, weight2 = result.x
    weight3 = 1 - weight1 - weight2
    normal1_pdf = norm.pdf(bins, mu1, sigma1) * weight1
    normal2_pdf = norm.pdf(bins, mu2, sigma2) * weight2
    normal3_pdf = norm.pdf(bins, mu3, sigma3) * weight3
    trimodal_pdf = trimodal(bins, mu1, sigma1, mu2, sigma2, mu3, sigma3, weight1, weight2)
    plt.plot(bins, trimodal_pdf, label='Fitted bimodal distribution', color='red', linewidth=1)
    plt.plot(bins, normal1_pdf, label='Normal distribution 1', color='green', linestyle='--', linewidth=1)
    plt.plot(bins, normal2_pdf, label='Normal distribution 2', color='blue', linestyle='--', linewidth=1)
    plt.plot(bins, normal3_pdf, label='Normal distribution 3', color='grey', linestyle='--', linewidth=1)

print(result.x)
plt.hist(x_log, BINS, density=True, stacked=True)

# plt.xscale('log')
# import pdb; pdb.set_trace()
# plt.xlim(0, 60)
if args.data == 'Flights':
    xlabel = f'log time (day)'
else:
    xlabel = f'log time (second)'
plt.xlabel(xlabel)
plt.legend()
plt.suptitle(f'{args.data} session length distribution')
plt.tight_layout()
if args.high_freq:
    plt.savefig(f'{args.data}_session_3.png')
else:
    plt.savefig(f'{args.data}_session.png')
# import pdb; pdb.set_trace()


# ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
# ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
# ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

# print('Sorting...')

# def tsort(i, indptr, indices, t, eid):
#     beg = indptr[i]
#     end = indptr[i + 1]
#     sidx = np.argsort(t[beg:end])
#     indices[beg:end] = indices[beg:end][sidx]
#     t[beg:end] = t[beg:end][sidx]
#     eid[beg:end] = eid[beg:end][sidx]



# for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
#     tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

# print('saving...')

# np.savez('DATA/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr,
#          indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)
