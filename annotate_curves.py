"""
Annotate a subset of the surprisal curves from get_autoregressive_surprisals.py.
Sample usage:

python3 lm-learning-curves/annotate_curves.py \
--training_file="datasets_tokenized_split/en_tokenized_train_shuffle0.txt" \
--sequences_file="datasets_tokenized_split/en_tokenized_eval_100000.txt" \
--vocab_size=50004 --model_dir="models/gpt2_0" \
--annotator_cache="annotators/gpt2_0" \
--surprisals_dir="surprisals/gpt2_0" --per_sequence=10 \
--compute_ngrams=True

"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from utils.annotator import CurveAnnotator
from utils.data_utils import get_checkpoints

def create_parser():
    parser = argparse.ArgumentParser()
    # Pre-training dataset for reference frequencies.
    parser.add_argument('--training_file', type=str, required=True)
    # Sequences file used when computing surprisals.
    parser.add_argument('--sequences_file', type=str, required=True)
    # Surprisals from get_autoregressive_surprisals.py.
    parser.add_argument('--surprisals_dir', type=str, required=True)
    # Where to save and load the annotator.
    parser.add_argument('--annotator_cache', type=str, required=True)
    # Model directory only used to get checkpoint steps.
    parser.add_argument('--model_dir', type=str, required=True)
    # Tokens to consider per sequence (e.g. 10 / 128).
    parser.add_argument('--per_sequence', type=int, default=10)
    # Vocab size to initialize n-gram counts and to compute chance surprisal.
    parser.add_argument('--vocab_size', type=int, default=50004)
    # Whether to run n-gram computations.
    parser.add_argument('--compute_ngrams', type=bool, default=False)
    return parser


def main(args):
    annotator = CurveAnnotator(args.annotator_cache)
    # Cache log10 steps in the annotator.
    checkpoint_steps = get_checkpoints(args.model_dir)
    log10_steps = np.log10(checkpoint_steps)
    annotator.set_log10_steps(log10_steps)
    # To select and cache the desired examples.
    examples = annotator.get_examples(args.sequences_file, per_sequence=args.per_sequence)
    del examples
    # To load and cache the surprisal curves.
    surprisal_curves = annotator.get_surprisal_curves(args.surprisals_dir)
    del surprisal_curves
    # Compute and cache the AoA values.
    chance_surprisal = -1.0 * np.log2(1.0 / args.vocab_size)
    aoa_values = annotator.get_aoa_values(chance_surprisal=chance_surprisal, proportion=0.50)
    del aoa_values
    # Compute GAM AoAs.
    gam_aoas = annotator.get_gam_aoas(chance_surprisal=chance_surprisal, proportion=0.50, gam_granularity=1000)
    del gam_aoas

    if args.compute_ngrams:
        # To compute and cache n-gram scores for the full training set.
        # Note: the cache for these files can be transferred for different training
        # runs, because the full train set stats are the same.
        # E.g. full_train_5gram_surprisals.npy and full_train_5gram_counts.pickle.
        # No pruning for n=1,2:
        for ngram_n in [1,2]:
            ngram_scores = annotator.get_ngram_surprisals('full_train', ngram_n=ngram_n,
                    sequences_path=args.sequences_file, reference_path=args.training_file,
                    reference_lines_mask=None, vocab_size=args.vocab_size, prune_every=999999999,
                    prune_minimum=None)
            del ngram_scores
        # Prune every 1M sequences with prune_minimum 2 for n=3,4,5:
        for ngram_n in [3,4,5]:
            ngram_scores = annotator.get_ngram_surprisals('full_train', ngram_n=ngram_n,
                    sequences_path=args.sequences_file, reference_path=args.training_file,
                    reference_lines_mask=None, vocab_size=args.vocab_size, prune_every=1000000,
                    prune_minimum=2)
            del ngram_scores

    # Get UMAP coordinates.
    # For different numbers of sample curves:
    # 10K: n_neighbors=5
    # 100K: n_neighbors=50
    # 1M: n_neighbors=500
    umap_coords = annotator.get_umap_coordinates(n_neighbors=500, n_components=2)
    del umap_coords
    # Fit GAMs.
    gam_curves = annotator.get_gam_curves()
    del gam_curves
    # Done.
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
