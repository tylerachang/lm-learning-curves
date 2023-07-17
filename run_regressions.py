"""
Run regressions and statistics.
"""
import os
import numpy as np
from tqdm import tqdm
import scipy
import statsmodels.api as sm

from utils.annotator import CurveAnnotator

# Predict the scores from context n-gram log-probability
# with different n-grams and window sizes.
# Scores shape: (n_runs, n_examples).
# Returns the maximum adjusted R^2.
def run_context_regressions(scores, annotator, ngrams, window_sizes):
    r2s = np.nan * np.ones((len(ngrams), len(window_sizes)))
    for ngram_i, ngram_n in enumerate(ngrams):
        for window_i, window_size in enumerate(window_sizes):
            context_logppls = annotator.get_context_ngram_logppls('full_train', ngram_n, window_size=window_size)
            x = np.concatenate([context_logppls]*scores.shape[0], axis=0)
            x = sm.add_constant(x.reshape(-1, 1))
            y = scores.flatten()
            reg = sm.OLS(y, x).fit()
            r2 = reg.rsquared_adj
            r2s[ngram_i, window_i] = r2
            print('Window {0}, ngram {1}: adjusted R^2 = {2}'.format(window_size, ngram_n, r2))
    print(r2s)
    max_r2 = np.max(r2s)
    return max_r2


def run_lrts(scores):
    r2 = 0
    model_null = sm.OLS(Y, X1)
    results_null = model_null.fit()
    model_alt = sm.OLS(Y, X2)
    results_alt = model_alt.fit()
    lr = -2.0 * (results_null.llf - results_alt.llf)
    df = results_alt.df_model - results_null.df_model
    p_value = scipy.stats.chi2.sf(lr, df)
    return r2


def main():
    annotators_dir = 'annotators'
    annotators = []
    for run_i in range(5):
        annotator = CurveAnnotator(os.path.join(annotators_dir, 'gpt2_{}'.format(run_i)))
        annotators.append(annotator)
    # Run regressions.

    # Predict scores from n-gram target log-probability.
    last_n = 11
    print('Predicting variability scores from context.')
    scores = []
    for annotator in annotators:
        scores.append(annotator.get_variability_scores(last_n=last_n))
    scores = np.array(scores)
    # Predict scores from context log-probability.
    run_context_regressions(scores, annotators[0], [1,2,3,4,5], [1, 3, 5, 10, 20, 100])

    # Predict scores from target 5-gram log-probability, unigram log-probability,
    # POS, and context log-probability (n-gram=..., window_size=...). Run
    # likelihood ratio tests for significance. Print R^2 for each individual
    # predictor, and overall regression.


if __name__ == "__main__":
    main()
