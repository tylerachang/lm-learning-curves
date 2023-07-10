import os
import numpy as np
from tqdm import tqdm
import codecs
from collections import defaultdict, Counter
import math
import pickle
import gc
from pygam import LinearGAM

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
get_ngram_surprisals()
get_ngram_surprisals_with_backoff()
get_context_lengths()
get_context_ngram_logppls()
get_target_ngram_surprisals()
get_curve_slopes()

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
            print('Using cached examples.')
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
            print('Using cached surprisal curves.')
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

    # Returns the confidence scores, the mean surprisal during the last_n checkpoints
    # of pre-training.
    # Shape: (n_examples).
    def get_confidence_scores(self, last_n=27):
        # Load surprisal curves or throw error.
        surprisal_curves_path = os.path.join(self.cache_dir, 'surprisal_curves.npy')
        if os.path.isfile(surprisal_curves_path):
            surprisal_curves = np.load(surprisal_curves_path, allow_pickle=False)
        else:
            print('Error: no surprisal curves cached; run get_surprisal_curves() first.')
        confidence_scores = np.mean(surprisal_curves[:, -last_n:], axis=-1)
        return confidence_scores

    # Returns the variability scores, the mean surprisal during the last_n checkpoints
    # of pre-training.
    # Shape: (n_examples).
    def get_variability_scores(self, last_n=27):
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
            print('Using cached AoA values.')
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
        assert tokens_mask.shape == ngram_surprisals.shape
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
    def get_gam_curves(self):
        # Load from cache if possible.
        gams_path = os.path.join(self.cache_dir, 'gam_curves.npy')
        if os.path.isfile(gams_path):
            print('Using cached GAM curves.')
            return np.load(gams_path, allow_pickle=False)
        print('Computing GAM curves.')
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
            # Note: uses 25 penalized b-splines.
            # Defaults to identity link and linear terms.
            gam = LinearGAM(n_splines=25)
            gam.gridsearch(X=log10_steps.reshape(-1, 1), y=surprisal_curves[example_i, :],
                    lam=np.logspace(-3, 3, 11, base=10.0), progress=False)
            predicted = gam.predict(log10_steps)
            gam_curves[curve_i, :] = predicted
        np.save(gams_path, gam_curves, allow_pickle=False)
        return gam_curves
