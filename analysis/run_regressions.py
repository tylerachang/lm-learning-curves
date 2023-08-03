"""
Run regressions and statistics.
"""
import os
import numpy as np
from tqdm import tqdm
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

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


def run_lrts(dataframe, target_var):
    print('Running regressions for target: {}'.format(target_var))
    all_predictors = ['unigram', 'ngram', 'context_loglen', 'context_logprob', 'contextual_div', 'pos']
    curr_predictors = []
    prev_reg = None
    for predictor in all_predictors:
        curr_predictors.append(predictor)
        formula = '{0} ~ {1} + 1'.format(target_var, ' + '.join(curr_predictors))
        curr_reg = smf.ols(formula=formula, data=dataframe).fit()
        print('  {}:'.format(predictor))
        if prev_reg is None:
            print('    R^2: {}'.format(curr_reg.rsquared_adj))
        else:
            likelihood_ratio = -2.0 * (prev_reg.llf - curr_reg.llf)
            df = curr_reg.df_model - prev_reg.df_model
            p_value = scipy.stats.chi2.sf(likelihood_ratio, df)
            print('    R^2: {}'.format(curr_reg.rsquared_adj))
            print('    (+{})'.format(curr_reg.rsquared_adj - prev_reg.rsquared_adj))
            print('    LRT p={}'.format(p_value))
        prev_reg = curr_reg
    # Add interactions.
    formula = '{0} ~ {1}'.format(target_var, ' + '.join(all_predictors))
    cont_predictors = [predictor for predictor in all_predictors if predictor != 'pos']
    pairs = itertools.combinations(cont_predictors, 2)
    for pred1, pred2 in pairs:
        formula += ' + {0}*{1}'.format(pred1, pred2)
    formula += ' + 1'
    # Interactions.
    curr_reg = smf.ols(formula=formula, data=dataframe).fit()
    likelihood_ratio = -2.0 * (prev_reg.llf - curr_reg.llf)
    df = curr_reg.df_model - prev_reg.df_model
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df)
    print('  Interactions:')
    print('    R^2: {}'.format(curr_reg.rsquared_adj))
    print('    (+{})'.format(curr_reg.rsquared_adj - prev_reg.rsquared_adj))
    print('    LRT p={}'.format(p_value))
    print('')
    return


def directions_of_effect(dataframe, target_var):
    print('Directions of effect for target: {}'.format(target_var))
    all_predictors = ['unigram', 'ngram', 'context_loglen', 'context_logprob', 'contextual_div', 'pos']
    # Run overall regression with all predictors.
    formula = '{0} ~ {1} + 1'.format(target_var, ' + '.join(all_predictors))
    overall_reg = smf.ols(formula=formula, data=dataframe).fit()
    # Get residuals from log-frequency regression.
    freq_reg = smf.ols(formula='{0} ~ unigram + 1'.format(target_var), data=dataframe).fit()
    freq_residuals = np.array(freq_reg.resid)
    # Get coefficients for each predictor in different regressions.
    cont_predictors = [predictor for predictor in all_predictors if predictor != 'pos']
    for predictor in cont_predictors:
        print_str = '  {}: '.format(predictor)
        # Sign in overall regression.
        sign = '+' if overall_reg.params[predictor] >= 0 else '-'
        print_str += sign
        # Sign in individual regression.
        reg = smf.ols(formula='{0} ~ {1} + 1'.format(target_var, predictor), data=dataframe).fit()
        sign = '+' if reg.params[predictor] >= 0 else '-'
        print_str += sign
        # Sign when predicting frequency residuals.
        if predictor != 'unigram':
            new_dataframe = pd.DataFrame()
            new_dataframe['predictor'] = dataframe[predictor]
            new_dataframe['resids'] = freq_residuals
            reg = smf.ols(formula='resids ~ predictor + 1', data=new_dataframe).fit()
            sign = '+' if reg.params['predictor'] >= 0 else '-'
            print_str += sign
        print(print_str)
    # POS results. Use coefficients after accounting for all other variables.
    formula = '{0} ~ {1} + 1'.format(target_var, ' + '.join(cont_predictors))
    cont_reg = smf.ols(formula=formula, data=dataframe).fit()
    residuals = np.array(cont_reg.resid)
    new_dataframe = pd.DataFrame()
    new_dataframe['pos'] = dataframe['pos']
    new_dataframe['resids'] = residuals
    pos_reg = smf.ols(formula='resids ~ pos + 1', data=new_dataframe).fit()
    pos_coefs = dict()
    for key in pos_reg.params.keys():
        if key.startswith('pos[T.'):
            pos_tag = key[6:-1]
            pos_coefs[pos_tag] = float(pos_reg.params[key])
    # Increasing coefficient order.
    sorted_tags = sorted(pos_coefs.keys(), key=pos_coefs.get)
    print('  POS tags: {}'.format(' '.join(sorted_tags)))
    print('')
    return


def main():
    annotators_dir = 'annotators'
    sequences_path = 'datasets_tokenized_split/en_tokenized_eval_100000.txt'
    annotators = []
    for run_i in range(5):
        annotator = CurveAnnotator(os.path.join(annotators_dir, 'gpt2_{}'.format(run_i)))
        annotators.append(annotator)
    average_df = get_average_features_dataframe(annotators, sequences_path)

    # Print predictor correlations.
    cont_predictors = ['unigram', 'ngram', 'context_loglen', 'context_logprob', 'contextual_div']
    print(average_df[cont_predictors].corr())
    print('')
    # Print VIFs.
    cont_df = average_df[cont_predictors]
    cont_df = add_constant(cont_df)
    vifs = pd.Series([variance_inflation_factor(cont_df.values, i) for i in range(cont_df.shape[1])],
                      index=cont_df.columns)
    print('VIFs:')
    print(vifs)
    print('')

    # Get directions of effects.
    for target_var in ['surprisal', 'var_steps', 'aoa', 'forgettability', 'var_runs']:
        directions_of_effect(average_df, target_var)

    # Run regressions and LRTs.
    for target_var in ['surprisal', 'var_steps', 'aoa', 'forgettability', 'var_runs']:
        run_lrts(average_df, target_var)

    # Predict scores from n-gram target log-probability.
    # print('Predicting variability scores from context.')
    # scores = np.array(dataframe['var_steps'])
    # Predict scores from context log-probability.
    # run_context_regressions(scores, annotators[0], [1,2,3,4,5], [1, 3, 5, 10, 20, 100])


if __name__ == "__main__":
    main()
