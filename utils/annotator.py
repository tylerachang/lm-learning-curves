import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import codecs
from collections import defaultdict, Counter
import math
import pickle
import gc
import itertools

from utils.data_utils import get_examples as get_file_examples
from utils.formula_utils import compute_aoa_vals, get_curve_slopes

"""
Class to get annotations for curves.
Curves to annotate are selected and the mask is cached during the first
get_examples() call. The remaining functions return annotations corresponding
to those curves. Some functions rely on the cache from other functions, but
errors will be printed if required caches are not found.

get_examples()
get_surprisal_curves()
get_confidence_scores()
get_variability_scores()
get_aoa_values()
get_gam_aoas()
get_ngram_surprisals()
get_ngram_surprisals_with_backoff()
get_context_lengths()
get_context_ngram_logppls()
get_target_ngram_surprisals()
get_gam_curves()
get_umap_coordinates()
get_pos_tag_sequences()
get_contextual_diversities()

"""
class CurveAnnotator:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # Sets and caches the log10_steps.
    def set_log10_steps(self, log10_steps):
        path = os.path.join(self.cache_dir, 'log10_steps.npy')
        np.save(path, log10_steps, allow_pickle=False)
    def get_log10_steps(self):
        path = os.path.join(self.cache_dir, 'log10_steps.npy')
        if os.path.isfile(path):
            return np.load(path, allow_pickle=False)
        else:
            print('log10_steps not cached; run set_log10_steps() first.')
            return None

    # Returns the selected examples for this annotator.
    # A list of lists of token ids (one list per example, ending in the token to predict).
    #
    # If cached: sequences_path is still required (because the tokens_mask is cached,
    # not the examples themselves).
    # If not cached: randomly selects examples based on sequences_path and per_example.
    # Caches the tokens_mask.
    def get_examples(self, sequences_path, per_sequence=None):
        # All the sequences. List of lists of token ids (one list per example).
        sequences = get_file_examples(sequences_path)
        # Assume all the same length.
        # Shape: (n_examples, seq_len).
        sequences = np.array(sequences)
        # Get examples mask if cached.
        tokens_mask_path = os.path.join(self.cache_dir, 'tokens_mask.npy')
        if os.path.isfile(tokens_mask_path):
            tokens_mask = np.load(tokens_mask_path, allow_pickle=False)
        else:
            print('Selecting examples.')
            if per_sequence is None:
                print('per_sequence must be set if examples are not cached.')
                return None
            tokens_mask = np.zeros(sequences.shape, dtype=bool)
            for sequence_i in range(sequences.shape[0]):
                # Cannot select the first index, because no surprisal for the
                # first token (because no previous token).
                indices = np.random.choice(sequences.shape[1]-1, size=per_sequence, replace=False)
                indices[:] += 1
                tokens_mask[sequence_i, indices] = True
            np.save(tokens_mask_path, tokens_mask, allow_pickle=False)
        assert tokens_mask.shape == sequences.shape
        # Examples.
        examples = []
        for sequence_i, sequence in enumerate(sequences):
            indices = np.where(tokens_mask[sequence_i, :])[0]
            for index in indices:
                # Sequence up to and including the index.
                examples.append(list(sequence[0:index+1]))
        print('Using {} examples.'.format(len(examples)))
        if per_sequence is not None:
            assert len(examples) == sequences.shape[0] * per_sequence
        return examples

    # Returns the surprisal curves.
    # Shape: (n_examples, n_checkpoints).
    def get_surprisal_curves(self, surprisals_dir=None):
        # Load from cache if possible.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            return np.load(surprisal_curves_path, allow_pickle=False)
        print('Loading surprisal curves.')
        # Load examples mask or throw error.
        tokens_mask_path = os.path.join(self.cache_dir, 'tokens_mask.npy')
        if os.path.isfile(tokens_mask_path):
            tokens_mask = np.load(tokens_mask_path, allow_pickle=False)
        else:
            print('Error: no examples selected; run get_examples() first.')
        # Get surprisal curve files.
        # Map filepath to (start_batch_index, end_batch_index).
        surprisal_batches = dict()
        for filename in os.listdir(surprisals_dir):
            if not filename.startswith('surprisals_tokens_'):
                continue
            if not filename.endswith('.npy'):
                continue
            # Exclude the prefix and suffix.
            batch_range = filename[18:-4]
            batch_range = batch_range.split('-')
            batch_start = int(batch_range[0])
            batch_end = int(batch_range[1])
            filepath = os.path.join(surprisals_dir, filename)
            surprisal_batches[filepath] = (batch_start, batch_end)
        sorted_filepaths = sorted(surprisal_batches.keys(),
                                  key = lambda key: surprisal_batches[key][0])
        # Loop through surprisal curve arrays, saving according to the mask.
        surprisal_curves = []
        # There is no surprisal for the first token of each sequence, because
        # there is no corresponding model prediction (because no previous token).
        tokens_mask = tokens_mask[:, 1:].flatten()
        curve_i = 0
        for filepath in tqdm(sorted_filepaths):
            # Shape: (batch_size, n_checkpoints).
            batch_curves = np.load(filepath, allow_pickle=False)
            # Check array shape.
            batch_start, batch_end = surprisal_batches[filepath]
            batch_size = batch_end - batch_start
            assert batch_curves.shape[0] == batch_size
            # Append the desired curves based on tokens_mask.
            for curve in batch_curves:
                if tokens_mask[curve_i]:
                    surprisal_curves.append(curve)
                curve_i += 1
        # Total curves should be equal to the mask length.
        assert curve_i == tokens_mask.shape[0]
        # Save and return.
        surprisal_curves = np.stack(surprisal_curves, axis=0)
        np.save(surprisal_curves_path, surprisal_curves, allow_pickle=False)
        return surprisal_curves

    # Returns the surprisal scores, the mean surprisal during the last_n checkpoints
    # of pre-training. Misnomer: high values = low confidence.
    # Shape: (n_examples).
    def get_confidence_scores(self, last_n=11):
        # Load surprisal curves or throw error.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            surprisal_curves = np.load(surprisal_curves_path, allow_pickle=False)
        else:
            print('Error: no surprisal curves cached; run get_surprisal_curves() first.')
        confidence_scores = np.mean(surprisal_curves[:, -last_n:], axis=-1)
        return confidence_scores

    # Returns the variability scores, the mean surprisal during the last_n checkpoints
    # of pre-training. These are within-run (across-step) variability scores.
    # Shape: (n_examples).
    def get_variability_scores(self, last_n=11):
        # Load surprisal curves or throw error.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            surprisal_curves = np.load(surprisal_curves_path, allow_pickle=False)
        else:
            print('Error: no surprisal curves cached; run get_surprisal_curves() first.')
        variability_scores = np.std(surprisal_curves[:, -last_n:], axis=-1)
        return variability_scores

    # Returns the age of acquisition scores, the step where a fitted sigmoid function
    # passes halfway between random chance and minimum surprisal. We use minimum
    # surprisal (best performance) instead of final confidence score as the
    # second baseline in case some curves increase then decrease in performance
    # during pre-training.
    # Also outputs the surprisal threshold and the fitted sigmoid parameters.
    # Overall output shape: (n_examples, 6).
    # Column 0: AoA (log-steps).
    # Column 1: surprisal threshold, or -1.0 if AoA was set to min or max steps.
    # Column 2-5: start, end, xmid, and scale for fitted sigmoid.
    # Note: sigmoids are usually decreasing (start > end) because surprisals
    # generally decrease.
    def get_aoa_values(self, chance_surprisal=None, proportion=0.50):
        # Load from cache if possible.
        aoa_path = os.path.join(self.cache_dir, 'aoa.npy')
        if os.path.isfile(aoa_path):
            return np.load(aoa_path, allow_pickle=False)
        print('Computing AoA values.')
        # Load surprisal curves or throw error.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            surprisal_curves = np.load(surprisal_curves_path, allow_pickle=False)
        else:
            print('Error: no surprisal curves cached; run get_surprisal_curves() first.')
        # Get corresponding log10 steps.
        log10_steps = self.get_log10_steps()
        if np.isinf(log10_steps[0]):
            # If first checkpoint step is zero, then log10 is -inf.
            log10_steps = log10_steps[1:]
            surprisal_curves = surprisal_curves[:, 1:]
        # Run AoA for each curve.
        n_curves = surprisal_curves.shape[0]
        all_aoa_vals = np.zeros((n_curves, 6))
        all_aoa_vals[:, :] = np.nan
        for curve_i in tqdm(range(n_curves)):
            aoa_vals = compute_aoa_vals(log10_steps, surprisal_curves[curve_i, :],
                    chance_surprisal, proportion=proportion)
            all_aoa_vals[curve_i, :] = np.array(aoa_vals)
        np.save(aoa_path, all_aoa_vals, allow_pickle=False)
        return all_aoa_vals

    # Get AoA scores, using fitted GAMs instead of sigmoids.
    # The first point on the GAM where the surprisal is less than or equal to
    # halfway between random chance and minimum surprisal.
    # Shape: (n_examples, 2).
    # Column 0: AoA (log-steps).
    # Column 1: surprisal threshold.
    def get_gam_aoas(self, chance_surprisal=None, proportion=0.50, gam_granularity=1000):
        # Load from cache if possible.
        gam_aoa_path = os.path.join(self.cache_dir, 'gam_aoa.npy')
        if os.path.isfile(gam_aoa_path):
            return np.load(gam_aoa_path, allow_pickle=False)
        print('Computing AoA values using GAMs.')
        from pygam import LinearGAM
        # Note: fitted GAMs are not loaded from the cache, because they are not
        # saved with high granularity.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            surprisal_curves = np.load(surprisal_curves_path, allow_pickle=False)
        else:
            print('Error: no surprisal curves cached; run get_surprisal_curves() first.')
        # Get corresponding log10 steps.
        log10_steps = self.get_log10_steps()
        if np.isinf(log10_steps[0]):
            log10_steps = log10_steps[1:]
            surprisal_curves = surprisal_curves[:, 1:]
        # For high granularity GAM.
        gam_x = np.linspace(log10_steps[0], log10_steps[-1], num=gam_granularity, endpoint=True)
        # Fit GAM and run AoA for each curve.
        n_curves = surprisal_curves.shape[0]
        gam_aoas = np.nan * np.ones((n_curves, 2))
        for curve_i, surprisal_curve in tqdm(enumerate(surprisal_curves)):
            # Same as in get_gam_curves().
            gam = LinearGAM(n_splines=25)
            gam.gridsearch(X=log10_steps.reshape(-1, 1), y=surprisal_curve,
                    lam=np.logspace(-3, 3, 11, base=10.0), progress=False)
            gam_curve = gam.predict(gam_x)
            # Get AoA.
            # Surprisals decreasing.
            ymax = chance_surprisal
            ymin = np.min(gam_curve)
            ythreshold = ymax*(1-proportion) + ymin*proportion
            gam_aoas[curve_i, 1] = ythreshold
            if ythreshold >= chance_surprisal:
                # Entire curve is above chance.
                gam_aoas[curve_i, 0] = log10_steps[-1]
                continue
            for step_i, log10_step in enumerate(gam_x):
                if gam_curve[step_i] <= ythreshold:
                    gam_aoas[curve_i, 0] = log10_step
                    break
        np.save(gam_aoa_path, gam_aoas, allow_pickle=False)
        return gam_aoas

    # For each sequence in sequences_path, returns the sequence of n-gram surprisals
    # conditioned on the previous tokens. For tokens with n-gram probability zero,
    # the surprisal is np.nan. Use get_ngram_surprisals_with_backoff() to
    # ensure no np.nans.
    #
    # Output shape: (n_sequences, seq_len).
    # Note: n_sequences is the number of input sequences, not the number of
    # selected examples (e.g. we select per_sequence individual token examples
    # per input sequence).
    # Note: reference_id allows for caching n-gram probabilities relative to
    # different reference texts.
    #
    # Note: when counting, this prunes counts less than prune_minimum every
    # prune_every sequences (and once at the end).
    # The final counts are cached.
    def get_ngram_surprisals(self, reference_id, ngram_n, sequences_path=None,
                             reference_path=None, reference_lines_mask=None,
                             vocab_size=None, prune_every=1000000,
                             prune_minimum=2):
        # Load from cache if possible.
        outpath = os.path.join(self.cache_dir, '{0}_{1}gram_surprisals.npy'.format(reference_id, ngram_n))
        if os.path.isfile(outpath):
            print('Using cached {}-gram surprisals.'.format(ngram_n))
            return np.load(outpath, allow_pickle=False)
        # Array of n-gram counts.
        # Entry i_0, ..., i_{n-1} is the count of
        # i_0, ..., i_{n-2}, i_{n-1}.
        # Dictionary mapping context tuples to Counters:
        # ngrams[(i-n+1, ..., i-1)][i] = count
        # Note: for unigrams, the first key is an empty tuple.
        ngrams_path = os.path.join(self.cache_dir, '{0}_{1}gram_counts.pickle'.format(reference_id, ngram_n))
        # Get ngram counts.
        if os.path.isfile(ngrams_path):
            print('Loading {}-gram counts.'.format(ngram_n))
            with open(ngrams_path, 'rb') as handle:
                ngrams = pickle.load(handle)
        else:
            print('Computing {}-gram counts.'.format(ngram_n))
            # Function to prune the ngrams dictionary.
            # Prunes anything with count 1.
            def prune_ngrams(ngrams, min_count=2):
                if min_count is None:
                    # No pruning.
                    return ngrams
                context_keys_to_remove = []
                for context, counts in ngrams.items():
                    target_keys_to_remove = []
                    for target, count in counts.items():
                        if count < min_count:
                            target_keys_to_remove.append(target)
                    for target in target_keys_to_remove:
                        counts.pop(target)
                    del target_keys_to_remove
                    # If all zero, prune this entire counter.
                    if len(counts) == 0:
                        context_keys_to_remove.append(context)
                for context in context_keys_to_remove:
                    ngrams.pop(context)
                # To resize the dictionary in memory after the removed keys.
                ngrams = ngrams.copy()
                del context_keys_to_remove
                gc.collect()
                return ngrams
            # Count ngrams. Create dictionary mapping:
            # ngrams[(i-n+1, ..., i-1)][i] = count
            # Note: for unigrams, the first key is an empty tuple.
            ngrams = defaultdict(lambda: Counter())
            reference_file = codecs.open(reference_path, 'rb', encoding='utf-8')
            line_count = 0
            for line_i, line in tqdm(enumerate(reference_file)):
                if reference_lines_mask and not reference_lines_mask[line_i]:
                    continue
                stripped_line = line.strip()
                if stripped_line == "":
                    continue
                sequence = [int(token_id) for token_id in stripped_line.split()]
                # Initialize with the extra pre-sequence tokens.
                # This represents the token_ids for the current ngram_n positions.
                curr = np.ones(ngram_n, dtype=int) * vocab_size
                for token_id in sequence:
                    # Increment to the next token.
                    curr = np.roll(curr, -1)
                    curr[-1] = token_id
                    # Increment the corresponding ngram:
                    ngrams[tuple(curr[:-1])][curr[-1]] += 1
                # Pruning.
                line_count += 1
                if line_count % prune_every == 0:
                    print('Pruning ngram counts <{}.'.format(prune_minimum))
                    orig_len = len(ngrams)
                    ngrams = prune_ngrams(ngrams, min_count=prune_minimum)
                    print('Pruned: {0} keys -> {1} keys.'.format(orig_len, len(ngrams)))
            print('Final prune: pruning ngram counts <{}.'.format(prune_minimum))
            orig_len = len(ngrams)
            ngrams = prune_ngrams(ngrams, min_count=prune_minimum)
            print('Pruned: {0} keys -> {1} keys.'.format(orig_len, len(ngrams)))
            # To allow pickling.
            ngrams.default_factory = None
            with open(ngrams_path, 'wb') as handle:
                pickle.dump(ngrams, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ngrams.default_factory = lambda: Counter()
        # Convert counts to conditional probabilities.
        # Entry i_0, ..., i_{n-1} is the probability of
        # i_{n-1} given i_0, ..., i_{n-2}.
        print('Converting counts to probabilities.')
        for context_key in ngrams:
            # Convert the counts to probabilities.
            counts = ngrams[context_key]
            total = np.sum(list(counts.values()))
            probs_dict = defaultdict(lambda: 0.0)
            for target_key, count in counts.items():
                prob = count / total
                probs_dict[target_key] = prob
            ngrams[context_key] = probs_dict

        # Get scores for all sequences.
        # Note: includes all examples, even those not included in get_examples().
        # This is because we want the surprisal for every token in every sequence,
        # to aggregate depending on each example and window size.
        # Surprisal is np.nan for n-grams with probability zero.
        print('Computing {}-gram surprisals.'.format(ngram_n))
        sequences = get_file_examples(sequences_path)
        # Assume all the same length.
        # Shape: (n_examples, seq_len).
        sequences = np.array(sequences)
        surprisals = -1.0 * np.ones(tuple(sequences.shape))
        for sequence_i, sequence in tqdm(enumerate(sequences)):
            # Fill previous tokens with placeholder.
            curr = np.ones(ngram_n, dtype=int) * vocab_size
            for token_i, token_id in enumerate(sequence):
                # Increment to the next token.
                curr = np.roll(curr, -1)
                curr[-1] = token_id
                # Increment the corresponding ngram:
                conditional_prob = ngrams[tuple(curr[:-1])][curr[-1]]
                if np.isclose(conditional_prob, 0.0):
                    surprisal = np.nan
                else:
                    surprisal = -1.0 * np.log2(conditional_prob)
                surprisals[sequence_i, token_i] = surprisal
        np.save(outpath, surprisals, allow_pickle=False)
        return surprisals

    # Returns n-gram suprisals as in get_ngram_surprisals(), but using backoff
    # to ensure no np.nans. For backoff, zero probabilities for ngram n are
    # replaced with the probabilities for ngram n-1. Unigram surprisals are
    # smoothed to the minimum nonzero probability.
    #
    # Assumes that get_ngram_surprisals() has already been run for all
    # 1 <= n <= ngram_n.
    def get_ngram_surprisals_with_backoff(self, reference_id, ngram_n):
        to_return = None
        has_nan = True
        curr_ngram_n = ngram_n
        while curr_ngram_n > 0 and has_nan:
            # Load from cache.
            inpath = os.path.join(self.cache_dir, '{0}_{1}gram_surprisals.npy'.format(reference_id, curr_ngram_n))
            if os.path.isfile(inpath):
                curr_ngrams = np.load(inpath, allow_pickle=False)
            else:
                print('Cannot find cached {}-gram surprisals; run get_ngram_surprisals() first.'.format(curr_ngram_n))
            # If unigrams, fill in with maximum surprisal (minimum nonzero probability).
            if curr_ngram_n == 1:
                max_surprisal = np.nanmax(curr_ngrams)
                curr_ngrams[np.isnan(curr_ngrams)] = max_surprisal
            # Fill in np.nans (backoff).
            if to_return is None:
                to_return = curr_ngrams
            else:
                nan_mask = np.isnan(to_return)
                to_return[nan_mask] = curr_ngrams[nan_mask]
            if np.sum(np.isnan(to_return)) == 0:
                has_nan = False
            # Decrement.
            curr_ngram_n -= 1
        return to_return

    # Returns the number of context tokens for each example.
    # Shape: (n_examples).
    # Note: as a predictor, np.log(context_lengths) tends to work better.
    def get_context_lengths(self):
        # Load examples mask or throw error.
        tokens_mask_path = os.path.join(self.cache_dir, 'tokens_mask.npy')
        if os.path.isfile(tokens_mask_path):
            tokens_mask = np.load(tokens_mask_path, allow_pickle=False)
        else:
            print('Error: no examples selected; run get_examples() first.')
        n_examples = np.sum(tokens_mask)
        # Collect context lengths.
        context_lengths = -1 * np.ones(n_examples, dtype=int)
        example_i = 0
        for sequence_i in range(tokens_mask.shape[0]):
            target_indices = np.argwhere(tokens_mask[sequence_i])[:, 0]
            for target_index in target_indices:
                # Note: target_index is equal to the number of preceding tokens
                # in the sequence, i.e. the context length.
                context_lengths[example_i] = target_index
                example_i += 1
        assert example_i == n_examples
        return context_lengths

    # Returns the cross-entropy (log perplexity) of each context for a given
    # window size, using an n-gram language model.
    # Shape: (n_examples).
    def get_context_ngram_logppls(self, reference_id, ngram_n, window_size=None):
        # Load examples mask or throw error.
        tokens_mask_path = os.path.join(self.cache_dir, 'tokens_mask.npy')
        if os.path.isfile(tokens_mask_path):
            tokens_mask = np.load(tokens_mask_path, allow_pickle=False)
        else:
            print('Error: no examples selected; run get_examples() first.')
        n_examples = np.sum(tokens_mask)
        # Load n-gram surprisals, for all n < ngram_n.
        # Shape: (ngram_n, n_examples, seq_len).
        ngram_surprisals = []
        for ngram_i in range(ngram_n):
            ngram_surprisals.append(self.get_ngram_surprisals_with_backoff(reference_id, ngram_i+1))
        ngram_surprisals = np.stack(ngram_surprisals, axis=0)
        assert tokens_mask.shape[0] == ngram_surprisals.shape[1]
        assert tokens_mask.shape[1] == ngram_surprisals.shape[2]
        # Compute context log perplexities. Equal to the mean surprisal of the context tokens.
        context_logppls = -1.0 * np.ones(n_examples)
        example_i = 0
        for sequence_i in tqdm(range(tokens_mask.shape[0])):
            target_indices = np.argwhere(tokens_mask[sequence_i])[:, 0]
            for target_index in target_indices:
                if window_size is None:
                    start_i = 0
                else:
                    start_i = target_index - window_size
                    start_i = start_i if start_i > 0 else 0
                # Compute log-probability of each context token, then mean.
                sum_logprob = 0.0
                for ngram_i in range(ngram_n-1):
                    if start_i+ngram_i >= target_index:
                        break
                    # For the first context tokens, use lower ngram.
                    sum_logprob += ngram_surprisals[ngram_i, sequence_i, start_i+ngram_i]
                if start_i+ngram_n-1 < target_index:
                    # Start using the ngram_n surprisal for context token ngram_n-1.
                    sum_logprob += np.sum(ngram_surprisals[-1, sequence_i, start_i+ngram_n-1:target_index])
                mean_logprob = sum_logprob / (target_index - start_i)
                context_logppls[example_i] = mean_logprob
                example_i += 1
        assert example_i == n_examples
        return context_logppls

    # Returns the cross-entropy of each target word,
    # using an n-gram language model.
    # Shape: (n_examples).
    def get_target_ngram_surprisals(self, reference_id, ngram_n):
        # Load examples mask or throw error.
        tokens_mask_path = os.path.join(self.cache_dir, 'tokens_mask.npy')
        if os.path.isfile(tokens_mask_path):
            tokens_mask = np.load(tokens_mask_path, allow_pickle=False)
        else:
            print('Error: no examples selected; run get_examples() first.')
        n_examples = np.sum(tokens_mask)
        # Load n-gram surprisals.
        ngram_surprisals = self.get_ngram_surprisals_with_backoff(reference_id, ngram_n)
        assert tokens_mask.shape == ngram_surprisals.shape
        # Compute target surprisal.
        target_surprisals = -1.0 * np.ones(n_examples)
        example_i = 0
        for sequence_i in tqdm(range(tokens_mask.shape[0])):
            target_indices = np.argwhere(tokens_mask[sequence_i])[:, 0]
            for target_index in target_indices:
                target_surprisal = ngram_surprisals[sequence_i, target_index]
                target_surprisals[example_i] = target_surprisal
                example_i += 1
        assert example_i == n_examples
        return target_surprisals

    # Deprecated. Not useful.
    def get_curve_slopes(self, window_size, stride):
        # Load from cache if possible.
        outpath = os.path.join(self.cache_dir, 'slopes_window{0}_stride{1}.npy'.format(window_size, stride))
        if os.path.isfile(outpath):
            return np.load(outpath, allow_pickle=False)
        # Load surprisal curves or throw error.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            surprisal_curves = np.load(surprisal_curves_path, allow_pickle=False)
        else:
            print('Error: no surprisal curves cached; run get_surprisal_curves() first.')
        n_curves = surprisal_curves.shape[0]
        # Get corresponding log10 steps.
        log10_steps = self.get_log10_steps()
        if np.isinf(log10_steps[0]):
            # If first checkpoint step is zero, then log10 is -inf.
            log10_steps = log10_steps[1:]
            surprisal_curves = surprisal_curves[:, 1:]
        # Compute slopes.
        # Compute one set of slopes to get shape.
        n_slopes = get_curve_slopes(log10_steps, surprisal_curves[0, :], window_size=window_size, stride=stride).shape[0]
        curve_slopes = np.nan * np.ones((n_curves, n_slopes))
        for curve_i in tqdm(range(n_curves)):
            slopes = get_curve_slopes(log10_steps, surprisal_curves[curve_i, :], window_size=window_size, stride=stride)
            curve_slopes[curve_i, :] = slopes
        np.save(outpath, curve_slopes, allow_pickle=False)
        return curve_slopes

    # Returns the GAM curves fitted to the surprisal curves.
    # Shape: (n_examples, n_checkpoints).
    # If first checkpoint is step 0, shape: (n_examples, n_checkpoints-1).
    def get_gam_curves(self, n_splines=-1):
        # Load from cache if possible.
        gams_path = os.path.join(self.cache_dir, 'gam_curves_{}splines.npy'.format(n_splines))
        if os.path.isfile(gams_path):
            return np.load(gams_path, allow_pickle=False)
        print('Computing GAM curves.')
        from pygam import LinearGAM
        # Load surprisal curves or throw error.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            surprisal_curves = np.load(surprisal_curves_path, allow_pickle=False)
        else:
            print('Error: no surprisal curves cached; run get_surprisal_curves() first.')
        # Get corresponding log10 steps.
        log10_steps = self.get_log10_steps()
        if np.isinf(log10_steps[0]):
            # If first checkpoint step is zero, then log10 is -inf.
            log10_steps = log10_steps[1:]
            surprisal_curves = surprisal_curves[:, 1:]
        # Fit GAM curve for each surprisal curve.
        n_curves = surprisal_curves.shape[0]
        gam_curves = np.zeros((n_curves, log10_steps.shape[0]))
        for curve_i in tqdm(range(n_curves)):
            # Note: uses n_splines penalized b-splines.
            # Defaults to identity link and linear terms.
            gam = LinearGAM(n_splines=n_splines)
            gam.gridsearch(X=log10_steps.reshape(-1, 1), y=surprisal_curves[curve_i, :],
                    lam=np.logspace(-3, 3, 11, base=10.0), progress=False)
            predicted = gam.predict(log10_steps)
            gam_curves[curve_i, :] = predicted
        np.save(gams_path, gam_curves, allow_pickle=False)
        return gam_curves

    # Returns the UMAP coordinates for the surprisal curves.
    # Shape: (n_examples, n_components).
    def get_umap_coordinates(self, n_neighbors=-1, n_components=2):
        # Load from cache if possible.
        umap_path = os.path.join(self.cache_dir, 'umap_coords_{0}d_{1}neighbors.npy'.format(n_components, n_neighbors))
        if os.path.isfile(umap_path):
            return np.load(umap_path, allow_pickle=False)
        print('Computing UMAP coordinates.')
        import umap
        # Load surprisal curves or throw error.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            surprisal_curves = np.load(surprisal_curves_path, allow_pickle=False)
        else:
            print('Error: no surprisal curves cached; run get_surprisal_curves() first.')
        # Run UMAP.
        umap_model = umap.UMAP(low_memory=False, n_neighbors=n_neighbors, metric='euclidean',
                               n_components=n_components, verbose=True)
        umap_coords = umap_model.fit_transform(surprisal_curves)
        np.save(umap_path, umap_coords, allow_pickle=False)
        return umap_coords

    # Assumes SentencePiece Hugging Face tokenizer.
    # Returns the POS tag sequence (list) for an input example (list of token_ids),
    # given a tokenizer and spaCy model. The POS tags are matched per token.
    def pos_tag_example(self, example, tokenizer, spacy_nlp):
        pos_tags = ['[CLS]']
        # Process a subsequence of the example (e.g. between [SEP] tokens).
        def process_sequence(sequence):
            # Map char idx to token idx.
            tokens = tokenizer.convert_ids_to_tokens(sequence)
            text = ''
            char_to_token_idx = []
            for token_idx, token in enumerate(tokens):
                if token_idx == 0:
                    # Skip new word character for first token.
                    token = token[1:]
                token = token.replace('\u2581', ' ')
                for char in token:
                    char_to_token_idx.append(token_idx)
                    text += char
            # Get and match POS tags.
            token_labels =  ['[UNMATCHED]']*len(sequence)
            spacy_sentence = spacy_nlp(text)
            for spacy_word in spacy_sentence:
                # Get the set of token indices that share characters with this spaCy word.
                start_char_idx = spacy_word.idx
                end_char_idx = start_char_idx + len(spacy_word)
                token_indices = set()
                for char_idx in range(start_char_idx, end_char_idx):
                    token_idx = char_to_token_idx[char_idx]
                    token_indices.add(token_idx)
                token_indices = [token_idx for token_idx in token_indices if token_labels[token_idx] == '[UNMATCHED]']
                # Set the corresponding token labels.
                # Appends _B (beginning), _I (inside), and _L (last).
                for token_i, token_idx in enumerate(sorted(token_indices)):
                    if len(token_indices) == 1:
                        suffix = '_U'
                    else:
                        suffix = '_B' if token_i == 0 else '_I'
                        if token_i == len(token_indices)-1:
                            suffix = '_L'
                    token_labels[token_idx] = spacy_word.pos_ + suffix
            pos_tags.extend(token_labels)
        # Tag all subsequences in the example.
        curr_sequence = []
        for token_id in example[1:]:
            if token_id == tokenizer.sep_token_id:
                process_sequence(curr_sequence)
                pos_tags.append('[SEP]')
                curr_sequence = []
            else:
                curr_sequence.append(token_id)
        if len(curr_sequence) > 0:
            process_sequence(curr_sequence)
        assert len(pos_tags) == len(example)
        # Enforce unknown character (<unk>) POS tags.
        for token_idx in np.argwhere(np.array(example) == tokenizer.unk_token_id)[:, 0]:
            pos_tags[token_idx] = '[UNK_CHAR]'
        return pos_tags

    # Return the list of POS tag sequences.
    # One list of POS tags per example.
    # You may first need to run: python3 -m spacy download [spacy_model_name]
    # Default spaCy model: en_core_web_sm
    def get_pos_tag_sequences(self, sequences_path=None, tokenizer=None):
        # Load from cache if possible.
        pos_path = os.path.join(self.cache_dir, 'pos_sequences.txt')
        if os.path.isfile(pos_path):
            tag_sequences = []
            infile = codecs.open(pos_path, 'rb', encoding='utf-8')
            for line in tqdm(infile):
                tag_sequences.append(line.strip().split())
            infile.close()
            return tag_sequences
        print('Tagging POS.')
        import spacy
        # Load examples.
        # This should have been run previously with per_sequence set.
        # Will throw an error otherwise.
        examples = self.get_examples(sequences_path, per_sequence=None)
        # Load spaCy. Only need the tagger and tok2vec for POS-tagging.
        spacy_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner',
                'entity_linker', 'entity_ruler', 'lemmatizer', 'textcat', 'textcat_multilabel',
                'senter', 'sentencizer', 'transformer'])
        tag_sequences = []
        for example in tqdm(examples):
            tag_sequence = self.pos_tag_example(example, tokenizer, spacy_nlp)
            tag_sequences.append(tag_sequence)
        # Save to cache.
        outfile = codecs.open(pos_path, 'wb', encoding='utf-8')
        for tag_sequence in tag_sequences:
            outfile.write(' '.join(tag_sequence))
            outfile.write('\n')
        outfile.close()
        return tag_sequences

    # Shape: (n_examples).
    # Forgettability scores.
    # Total height of increasing sections in smoothed GAM curve. Equivalent to
    # the total difference between each relative minimum and its corresponding
    # next relative maximum. As long as the curve is the same, independent of
    # x-scale (i.e. log vs. standard).
    # Use a smoothed curve, because otherwise increased noise would lead to
    # many irrelevant increases in surprisal. Each spike and spurious change
    # would contribute to the forgettability score. This is better captured
    # by variability, although the two are related (e.g. high variability
    # might indicate some type of forgettability).
    def get_forgettability_scores(self):
        gam_curves = self.get_gam_curves(n_splines=25)
        # Positive if increasing, negative if decreasing.
        changes = gam_curves[:, 1:] - gam_curves[:, :-1]
        # Shape: (n_examples, n_checkpoints-1).
        changes = np.clip(changes, a_min=0.0, a_max=None)
        forgettability = np.sum(changes, axis=-1)
        return forgettability

    # Shape: (n_examples).
    # Gets the contextual diversity per example.
    # Each value is the count of unique previous tokens.
    def get_contextual_diversities(self, reference_id, window_size=30,
                                   most_frequent=10000, sequences_path=None,
                                   reference_path=None, vocab_size=None,
                                   max_sequences=-1, adjusted=False):
        # Load per-token contextual diversities from cache if possible.
        diversities_path = '{0}_contextual_diversities_{1}window_{2}frequent.npy'.format(reference_id, window_size, most_frequent)
        if adjusted:
            # Need to have run adjust_contextual_diversities().
            diversities_path = diversities_path.replace('.npy', '_adjusted.npy')
        diversities_path = os.path.join(self.cache_dir, diversities_path)
        if os.path.isfile(diversities_path):
            # Shape: (vocab_size).
            diversities = np.load(diversities_path, allow_pickle=False)
        else:
            print('Computing raw contextual diversities.')
            assert window_size > 0, 'Window size must be > 0.'
            def update_co_occurrences(occurrence_matrix, example, window_size):
                # Update the co-occurrence matrix with a single example.
                # Rows: target_tokens, columns: co-occurring tokens.
                # Shape: (vocab_size, n_tokens_to_consider). Considers
                # co-occurrences with the first most_frequent tokens.
                # example should be a list of token ids.
                for idx, id in enumerate(example):
                    # window_size previous tokens.
                    start_window = max(0, idx-window_size)
                    end_window = idx # Range is exclusive of end_window.
                    for idx2 in range(start_window, end_window):
                        id2 = example[idx2]
                        if id2 < occurrence_matrix.shape[1]:
                            occurrence_matrix[id, id2] += 1
                return occurrence_matrix
            # Count lines in infile, to sample the desired number of sequences.
            print('Counting lines in reference file.')
            infile = codecs.open(reference_path, 'rb', encoding='utf-8')
            total_lines = 0
            for line in tqdm(infile):
                total_lines += 1
            infile.close()
            if max_sequences != -1 and total_lines > max_sequences:
                lines_mask = np.zeros(total_lines, dtype=bool)
                indices = np.random.choice(total_lines, size=max_sequences, replace=False)
                lines_mask[indices] = True
            else:
                lines_mask = np.ones(total_lines, dtype=bool)
            # Compute co-occurrence matrix.
            occurrences = np.zeros((vocab_size, most_frequent), dtype=np.int32)
            token_count = 0
            infile = codecs.open(reference_path, 'rb', encoding='utf-8')
            for line_i, line in tqdm(enumerate(infile), total=total_lines):
                if not lines_mask[line_i]:
                    continue
                example = [int(token_id) for token_id in line.strip().split()]
                token_count += len(example)
                occurrences = update_co_occurrences(occurrences, example, window_size)
            infile.close()
            print('Total token count: {}'.format(token_count))
            # For each token_id (row), how many unique tokens it co-occurs with.
            diversities = np.sum(occurrences > 0, axis=-1)
            np.save(diversities_path, diversities, allow_pickle=False)
            print('Saved raw contextual diversities.')
        # Map each example to the contextual diversity for the target token.
        # Load examples.
        # This should have been run previously with per_sequence set.
        # Will throw an error otherwise.
        examples = self.get_examples(sequences_path, per_sequence=None)
        target_ids = [example[-1] for example in examples]
        del examples
        example_diversities = [diversities[target_id] for target_id in target_ids]
        return np.array(example_diversities)

    # Adjusts and saves contextual_diversities.
    # The GAM curve is fitted without the CLS token (very high frequency, zero
    # raw diversity). The CLS token adjusted diversity is set to the minimum
    # adjusted diversity of any other token.
    # Must have run get_contextual_diversities() and get_ngram_surprisals() for
    # unigrams.
    def adjust_contextual_diversities(self, reference_id, window_size=30,
            most_frequent=10000, vocab_size=None, cls_token=None,
            unigram_reference_id='full_train', n_splines=25):
        # Load per-token contextual diversities from cache if possible.
        diversities_path = '{0}_contextual_diversities_{1}window_{2}frequent.npy'.format(reference_id, window_size, most_frequent)
        diversities_path = os.path.join(self.cache_dir, diversities_path)
        adjusted_diversities_path = diversities_path.replace('.npy', '_adjusted.npy')
        if os.path.isfile(adjusted_diversities_path):
            print('Contextual diversities already adjusted.')
            return
        # Shape: (vocab_size).
        diversities = np.load(diversities_path, allow_pickle=False)
        # Mask CLS token.
        token_mask = np.ones(vocab_size, dtype=bool)
        token_mask[cls_token] = False  # CLS token has high frequency, zero diversity.
        # Get unigram log-frequencies.
        ngrams_path = os.path.join(self.cache_dir, '{}_1gram_counts.pickle'.format(unigram_reference_id))
        with open(ngrams_path, 'rb') as handle:
            ngrams = pickle.load(handle)
        unigram_frequencies = np.nan * np.ones(vocab_size)
        unigram_counts = ngrams[tuple()]
        total_count = np.sum(list(unigram_counts.values()))
        for token_id, count in unigram_counts.items():
            unigram_frequencies[token_id] = count / total_count
        # Normalize to minimum frequency to remove -inf logs.
        unigram_frequencies[np.isnan(unigram_frequencies)] = np.nanmin(unigram_frequencies)
        logfreqs = np.log10(unigram_frequencies)
        # Mask.
        diversities = diversities[token_mask]
        logfreqs = logfreqs[token_mask]
        # Compute GAM.
        from pygam import LinearGAM
        gam = LinearGAM(n_splines=n_splines)
        gam.gridsearch(X=logfreqs.reshape(-1, 1), y=diversities,
                lam=np.logspace(-3, 3, 11, base=10.0), progress=False)
        predicted = gam.predict(logfreqs)
        # Adjusted diversities.
        adj_diversities = np.nan * np.ones(vocab_size)
        adj_diversities[token_mask] = diversities - predicted
        adj_diversities[cls_token] = np.nanmin(adj_diversities)
        np.save(adjusted_diversities_path, adj_diversities, allow_pickle=False)
        return adj_diversities


    # Shape: (n_examples).
    # Noise variability.
    # Distance between surprisal curve and fitted GAM with 25 splines.
    def get_noise_scores(self):
        # Not-implemented yet.
        return None

    # Shape: (n_examples).
    # Fluctuation variability.
    # Distance between fitted GAMs with 25 vs. 5 splines.
    def get_fluctuation_scores(self):
        # Not-implemented yet.
        return None


"""
Functions that aggregate across annotators.
"""

# Shape: (n_examples, n_pairwise).
# Distance between fitted GAMs with 25 splines, across runs.
# Each column is a pair of pre-training runs.
# Should capture general trend, so using GAMs excludes noise.
# For each pair of curves, uses squared Euclidean distance divided by n_checkpoints
# (i.e. the mean squared difference between the two curves).
def get_crossrun_variability(annotators, use_gams=True):
    # Load from cache if possible.
    if use_gams:
        outpath = 'crossrun_distances_gams.npy'
    else:
        outpath = 'crossrun_distances_raw.npy'
    if os.path.isfile(outpath):
        return np.load(outpath, allow_pickle=False)
    print('Computing cross-run distances.')
    all_curves = []
    for annotator in annotators:
        if use_gams:
            curves = annotator.get_gam_curves(n_splines=25)
        else:
            curves = annotator.get_surprisal_curves()
        all_curves.append(curves)
    n_examples = all_curves[0].shape[0]
    pairs = list(itertools.combinations(range(len(annotators)), 2))
    crossrun_distances = np.nan * np.ones((n_examples, len(pairs)))
    for pair_i, pair in enumerate(pairs):
        i, j = pair
        # Shape: (n_examples).
        dists = np.mean(np.square(all_curves[i] - all_curves[j]), axis=-1)
        crossrun_distances[:, pair_i] = dists
    np.save(outpath, crossrun_distances, allow_pickle=False)
    return crossrun_distances


# Get a DataFrame of surprisal, variability (within-run), AoA, forgettability, unigram
# target surprisal, 5-gram target surprisal, context unigram log-perplexity, and
# POS. Includes a separate entry for each pre-training run score. Includes
# contextual diversity if sequences_path is included.
def get_features_dataframe(annotators, sequences_path=None):
    full_dataframe = pd.DataFrame()
    # These are independent of the pre-training run.
    unigram_scores = annotators[0].get_target_ngram_surprisals('full_train', 1)
    ngram_scores = annotators[0].get_target_ngram_surprisals('full_train', 5)
    context_scores = annotators[0].get_context_ngram_logppls('full_train', ngram_n=1, window_size=None)
    diversity_scores = None
    if sequences_path is not None:
        diversity_scores = annotators[0].get_contextual_diversities('train1b', window_size=30,
                most_frequent=10000, sequences_path=sequences_path, adjusted=True)
    pos_tags = [sequence[-1] for sequence in annotators[0].get_pos_tag_sequences()]
    # For cross-run variability, use the average cross-run distance, across
    # pre-training run pairs.
    var_runs = np.mean(get_crossrun_variability(annotators, use_gams=True), axis=-1)
    for annotator_i, annotator in enumerate(annotators):
        data = dict()
        data['unigram'] = unigram_scores
        data['ngram'] = ngram_scores
        data['context'] = context_scores
        if diversity_scores is not None:
            data['contextual_div'] = diversity_scores
        data['pos'] = pos_tags
        data['var_runs'] = var_runs
        data['surprisal'] = annotator.get_confidence_scores(last_n=11)
        data['var_steps'] = annotator.get_variability_scores(last_n=11)
        data['aoa'] = annotator.get_gam_aoas()[:, 0]
        data['forgettability'] = annotator.get_forgettability_scores()
        data['run_i'] = np.ones(len(unigram_scores), dtype=int) * annotator_i
        data['example_i'] = np.arange(0, len(unigram_scores), dtype=int)
        dataframe = pd.DataFrame(data)
        dataframe['pos'] = dataframe['pos'].astype('category')
        full_dataframe = pd.concat([full_dataframe, dataframe], ignore_index=True, axis=0)
    return full_dataframe


# Get a DataFrame of features, averaged across pre-training runs.
# Also includes cross-run variability.
def get_average_features_dataframe(annotators, sequences_path):
    df = get_features_dataframe(annotators, sequences_path=sequences_path)
    pos = df['pos']  # Cannot mean over categorical column.
    # Group by example_i, mean, re-index.
    df = df.drop(columns=['pos', 'run_i']).groupby('example_i').mean().reset_index()
    # Add pos back in.
    df['pos'] = pos
    # Add cross-run variability.
    crossrun_var = np.mean(get_crossrun_variability(annotators, use_gams=True), axis=-1)
    df['var_runs'] = crossrun_var
    return df
