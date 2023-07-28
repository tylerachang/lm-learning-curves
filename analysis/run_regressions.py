"""
Run regressions and statistics.
"""
import os
import numpy as np
from tqdm import tqdm
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sys
sys.path.append('lm-learning-curves')
from utils.annotator import CurveAnnotator, get_features_dataframe, get_average_features_dataframe

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


def run_lrts(dataframe, target_var, crossrun_r2=1.0):
    print('Running regressions for target: {}'.format(target_var))
    # Unigram.
    unigram_reg = smf.ols(formula='{} ~ unigram + 1'.format(target_var), data=dataframe).fit()
    print('  Unigram R^2: {}'.format(unigram_reg.rsquared_adj))
    # 5-gram.
    ngram_reg = smf.ols(formula='{} ~ unigram + ngram + 1'.format(target_var), data=dataframe).fit()
    likelihood_ratio = -2.0 * (unigram_reg.llf - ngram_reg.llf)
    df = ngram_reg.df_model - unigram_reg.df_model
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df)
    print('  Unigram+ngram R^2: {}'.format(ngram_reg.rsquared_adj))
    print('    LRT significance: p={}'.format(p_value))
    print('    Improvement: +{}'.format(ngram_reg.rsquared_adj - unigram_reg.rsquared_adj))
    # Context.
    context_reg = smf.ols(formula='{} ~ unigram + ngram + context + 1'.format(target_var), data=dataframe).fit()
    likelihood_ratio = -2.0 * (ngram_reg.llf - context_reg.llf)
    df = context_reg.df_model - ngram_reg.df_model
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df)
    print('  Unigram+ngram+context R^2: {}'.format(context_reg.rsquared_adj))
    print('    LRT significance: p={}'.format(p_value))
    print('    Improvement: +{}'.format(context_reg.rsquared_adj - ngram_reg.rsquared_adj))
    # Contextual diversity.
    contextdiv_reg = smf.ols(formula='{} ~ unigram + ngram + context + contextual_div + 1'.format(target_var), data=dataframe).fit()
    likelihood_ratio = -2.0 * (context_reg.llf - contextdiv_reg.llf)
    df = contextdiv_reg.df_model - context_reg.df_model
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df)
    print('  Unigram+ngram+context+contextdiv R^2: {}'.format(contextdiv_reg.rsquared_adj))
    print('    LRT significance: p={}'.format(p_value))
    print('    Improvement: +{}'.format(contextdiv_reg.rsquared_adj - context_reg.rsquared_adj))
    # Interactions.
    interactions_reg = smf.ols(formula='{} ~ unigram + ngram + context + unigram*ngram + unigram*context + ngram*context + 1'.format(target_var), data=dataframe).fit()
    likelihood_ratio = -2.0 * (context_reg.llf - interactions_reg.llf)
    df = interactions_reg.df_model - context_reg.df_model
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df)
    print('  Unigram+ngram+context+interactions R^2: {}'.format(interactions_reg.rsquared_adj))
    print('    LRT significance: p={}'.format(p_value))
    # POS.
    pos_reg = smf.ols(formula='{} ~ unigram + ngram + context + contextual_div + pos + 1'.format(target_var), data=dataframe).fit()
    likelihood_ratio = -2.0 * (contextdiv_reg.llf - pos_reg.llf)
    df = pos_reg.df_model - contextdiv_reg.df_model
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df)
    print('  Unigram+ngram+context+POS R^2: {}'.format(pos_reg.rsquared_adj))
    print('    LRT significance: p={}'.format(p_value))
    print('    Improvement (excluding interactions): +{}'.format(pos_reg.rsquared_adj - contextdiv_reg.rsquared_adj))
    # print('')
    # print(pos_reg.summary())
    print('')
    return


def main():
    annotators_dir = 'annotators'
    sequences_path = 'datasets_tokenized_split/en_tokenized_eval_100000.txt'
    annotators = []
    for run_i in range(5):
        annotator = CurveAnnotator(os.path.join(annotators_dir, 'gpt2_{}'.format(run_i)))
        annotators.append(annotator)

    # Run regressions and LRTs.
    average_df = get_average_features_dataframe(annotators, sequences_path)
    for target_var in ['surprisal', 'var_steps', 'aoa', 'forgettability', 'var_runs']:
        crossrun_r2 = np.square(crossrun_r[target_var])
        run_lrts(average_df, target_var, crossrun_r2=crossrun_r2)

    # Predict scores from n-gram target log-probability.
    # print('Predicting variability scores from context.')
    # scores = np.array(dataframe['var_steps'])
    # Predict scores from context log-probability.
    # run_context_regressions(scores, annotators[0], [1,2,3,4,5], [1, 3, 5, 10, 20, 100])


if __name__ == "__main__":
    main()
