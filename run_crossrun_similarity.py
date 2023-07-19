"""
Are general trajectories similar across pre-training runs?
For each curve, what is the rank of its corresponding curve in another
pre-training run, when examples are sorted by curve distance?
"""
import os
import numpy as np
from tqdm import tqdm
import itertools
import multiprocessing as mp

from utils.annotator import CurveAnnotator

USE_GAMS = False
RUN_I = 0
RUN_J = 1

filepath = 'rank_similarities/run{0}_run{1}_similarity_ranks'.format(RUN_I, RUN_J)
if USE_GAMS:
    filepath += '_gams.npy'
else:
    filepath += '_raw.npy'
if os.path.isfile(filepath):
    print('WARNING: already run.')

annotators_dir = 'annotators'
annotator1 = CurveAnnotator(os.path.join(annotators_dir, 'gpt2_{}'.format(RUN_I)))
annotator2 = CurveAnnotator(os.path.join(annotators_dir, 'gpt2_{}'.format(RUN_J)))
# Compute distances from either GAMs or raw curves.
curves1 = None
curves2 = None
if USE_GAMS:
    curves1 = annotator1.get_gam_curves(n_splines=25)
    curves2 = annotator2.get_gam_curves(n_splines=25)
else:
    curves1 = annotator1.get_surprisal_curves()
    curves2 = annotator2.get_surprisal_curves()

# Returns a (n_examples) integer array.
# Each entry is the rank of the corresponding curve in curves2.
#
# First, function to get the rank for example_i.
def get_rank(example_i):
    global curves1
    global curves2
    curve1 = curves1[example_i, :]
    # Shape: (n_examples).
    dists = np.mean(np.square(curve1.reshape(1, -1) - curves2), axis=-1)
    # Distance to the corresponding curve in annotator2.
    correct_dist = dists[example_i]
    # Rank: number of examples that are less than the correct_dist.
    # Best rank: 0
    # Worst value: n_examples-1
    rank = np.sum(dists < correct_dist)
    return rank
def get_trajectory_ranks():
    # How many cores?
    n_cpus = mp.cpu_count()
    n_cpus = 4  # This is actually faster in some cases.
    # Compute ranks.
    global curves1
    n_examples = curves1.shape[0]
    pool = mp.Pool(processes=n_cpus)
    jobs = []
    print("Using {} processes.".format(n_cpus))
    for example_i in range(n_examples): # Add jobs.
        job = pool.apply_async(get_rank, args=(example_i,))
        jobs.append(job)
    pool.close()
    for job in tqdm(jobs): # Progress bar.
        job.wait()
    results = [job.get() for job in jobs]
    results = np.array(results)
    return results

ranks = get_trajectory_ranks()
np.save(filepath, ranks, allow_pickle=False)
