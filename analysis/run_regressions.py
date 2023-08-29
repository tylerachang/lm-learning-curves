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

# Predict from context n-gram log-probability with different n-grams and window sizes.
# Prints the R2 scores.
# Prints any directions of effect in an unexpected direction.
def run_context_regressions(dataframe, target_var, annotator, ngrams, window_sizes, expected_sign='-'):
    print('Context log-prob effects for target: {}'.format(target_var))
    all_predictors = ['unigram', 'ngram', 'context_loglen', 'context_logprob', 'contextual_div', 'pos', 'pos_bilu']
    original_context = dataframe['context_logprob']  # To reset at the end.
    # Get residuals from log-frequency regression.
    freq_reg = smf.ols(formula='{0} ~ unigram + 1'.format(target_var), data=dataframe).fit()
    freq_residuals = np.array(freq_reg.resid)
    # Run regressions.
    r2s = np.nan * np.ones((len(ngrams), len(window_sizes)))
    n_warnings = 0
    for ngram_i, ngram_n in enumerate(ngrams):
        for window_i, window_size in enumerate(window_sizes):
            if window_size < ngram_n:
                continue
            context_logppls = annotator.get_context_ngram_logppls('full_train', ngram_n, window_size=window_size)
            dataframe['context_logprob'] = -1.0 * context_logppls
            # Sign in overall regression with all predictors.
            formula = '{0} ~ {1} + 1'.format(target_var, ' + '.join(all_predictors))
            overall_reg = smf.ols(formula=formula, data=dataframe).fit()
            sign = '+' if overall_reg.params['context_logprob'] >= 0 else '-'
            if sign != expected_sign:
                print('WARNING: window {0}, ngram {1}, overall regression: {2}'.format(window_size, ngram_n, sign))
                n_warnings += 1
            # Sign in individual regression.
            reg = smf.ols(formula='{0} ~ {1} + 1'.format(target_var, 'context_logprob'), data=dataframe).fit()
            sign = '+' if reg.params['context_logprob'] >= 0 else '-'
            if sign != expected_sign:
                print('WARNING: window {0}, ngram {1}, individual regression: {2}'.format(window_size, ngram_n, sign))
                n_warnings += 1
            # Save this R2.
            r2s[ngram_i, window_i] = reg.rsquared_adj
            # As an indicator when printing, use the sign from the individual regression.
            if sign == '-': r2s[ngram_i, window_i] *= -1.0
            # Sign when predicting frequency residuals.
            new_dataframe = pd.DataFrame()
            new_dataframe['context_logprob'] = dataframe['context_logprob']
            new_dataframe['resids'] = freq_residuals
            reg = smf.ols(formula='resids ~ context_logprob + 1', data=new_dataframe).fit()
            sign = '+' if reg.params['context_logprob'] >= 0 else '-'
            if sign != expected_sign:
                print('WARNING: window {0}, ngram {1}, freq-adjusted individual regression: {2}'.format(window_size, ngram_n, sign))
                n_warnings += 1
    dataframe['context_logprob'] = original_context  # Reset dataframe.
    print('R^2s:')
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    print(r2s)
    max_r2 = np.nanmax(np.absolute(r2s))
    print('Max R^2: {}'.format(max_r2))
    print('Unexpected directions: {}'.format(n_warnings))
    return


def run_lrts(dataframe, target_var):
    print('Running regressions for target: {}'.format(target_var))
    all_predictors = ['unigram', 'ngram', 'context_loglen', 'context_logprob', 'contextual_div', ['pos', 'pos_bilu']]
    curr_predictors = []
    prev_reg = None
    for predictor in all_predictors:
        if isinstance(predictor, list):
            curr_predictors.extend(predictor)
        else:
            curr_predictors.append(predictor)
        formula = '{0} ~ {1} + 1'.format(target_var, ' + '.join(curr_predictors))
        curr_reg = smf.ols(formula=formula, data=dataframe).fit()
        print('  {}:'.format(str(predictor)))
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
    formula = '{0} ~ {1}'.format(target_var, ' + '.join(curr_predictors))
    cont_predictors = [predictor for predictor in curr_predictors if predictor not in ['pos', 'pos_bilu']]
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


def directions_of_effect(dataframe, target_var, table_format=False):
    print('Directions of effect for target: {}'.format(target_var))
    missing = 'raise'  # Strategy for missing or nan data.
    all_predictors = ['unigram', 'ngram', 'context_loglen', 'context_logprob', 'contextual_div', 'pos', 'pos_bilu']
    # Run overall regression with all predictors.
    formula = '{0} ~ {1} + 1'.format(target_var, ' + '.join(all_predictors))
    overall_reg = smf.ols(formula=formula, data=dataframe, missing=missing).fit()
    # Get residuals from log-frequency regression.
    freq_reg = smf.ols(formula='{0} ~ unigram + 1'.format(target_var), data=dataframe, missing=missing).fit()
    freq_residuals = np.nan * np.ones_like(dataframe[target_var])
    freq_residuals[np.logical_not(np.isnan(dataframe[target_var]))] = np.array(freq_reg.resid)
    # Get coefficients for each predictor in different regressions.
    cont_predictors = [predictor for predictor in all_predictors if predictor not in ['pos', 'pos_bilu']]
    for predictor in cont_predictors:
        print_str = '  {}: '.format(predictor)
        # Sign in overall regression.
        sign = '+' if overall_reg.params[predictor] >= 0 else '-'
        print_str += sign
        # Sign in individual regression.
        reg = smf.ols(formula='{0} ~ {1} + 1'.format(target_var, predictor), data=dataframe, missing=missing).fit()
        sign = '+' if reg.params[predictor] >= 0 else '-'
        print_str += sign
        # Sign when predicting frequency residuals.
        if predictor != 'unigram':
            new_dataframe = pd.DataFrame()
            new_dataframe['predictor'] = dataframe[predictor]
            new_dataframe['resids'] = freq_residuals
            reg = smf.ols(formula='resids ~ predictor + 1', data=new_dataframe, missing=missing).fit()
            sign = '+' if reg.params['predictor'] >= 0 else '-'
            print_str += sign
        print(print_str)
    # POS results. Use coefficients after accounting for all other variables.
    formula = '{0} ~ {1} + 1'.format(target_var, ' + '.join(cont_predictors))
    cont_reg = smf.ols(formula=formula, data=dataframe, missing=missing).fit()
    residuals = np.array(cont_reg.resid)
    # Find R2 increase from adding just POS vs. just BILU.
    cont_reg_r2 = cont_reg.rsquared_adj
    pos_reg = smf.ols(formula = formula + ' + pos', data=dataframe, missing=missing).fit()
    r2_increase = pos_reg.rsquared_adj - cont_reg_r2
    print('R2 increase from POS alone (no BILU): {}'.format(r2_increase))
    bilu_reg = smf.ols(formula = formula + ' + pos_bilu', data=dataframe, missing=missing).fit()
    r2_increase = bilu_reg.rsquared_adj - cont_reg_r2
    print('R2 increase from BILU alone: {}'.format(r2_increase))
    # Get coefficients.
    new_dataframe = pd.DataFrame()
    new_dataframe['pos'] = dataframe['pos']
    new_dataframe['pos_bilu'] = dataframe['pos_bilu']
    new_dataframe['resids'] = residuals
    formula = 'resids ~ C(pos, Treatment(reference="[UNMATCHED]")) + C(pos_bilu, Treatment(reference="U")) + 1'
    pos_reg = smf.ols(formula=formula, data=new_dataframe, missing=missing).fit()
    # Get coefs for POS and BILU tags.
    pos_coefs = dict()
    pos_coefs['[UNMATCHED]'] = 0.0
    bilu_coefs = dict()
    bilu_coefs['U'] = 0.0
    # print(list(pos_reg.params.keys()))
    for key in pos_reg.params.keys():
        if key.startswith('C(pos, Treatment(reference="[UNMATCHED]"))[T.'):
            pos_tag = key[45:-1]
            pos_coefs[pos_tag] = float(pos_reg.params[key])
        elif key.startswith('C(pos_bilu, Treatment(reference="U"))[T.'):
            bilu_tag = key[40:-1]
            bilu_coefs[bilu_tag] = float(pos_reg.params[key])
    # Increasing coefficient order.
    sorted_pos = sorted(pos_coefs.keys(), key=pos_coefs.get)
    sorted_bilu = sorted(bilu_coefs.keys(), key=bilu_coefs.get)
    if table_format:
        # Format POS coefficients as table.
        print('  Tag & Coef. \\\\')
        for pos_tag in sorted_pos:
            str_tag = pos_tag.replace('[', '{[}')
            str_tag = str_tag.replace(']', '{]}')
            print('{0} & {1} \\\\'.format(str_tag, round(pos_coefs[pos_tag], 2)))
        print('  Tag & Coef. \\\\')
        for bilu_tag in sorted_bilu:
            str_tag = bilu_tag.replace('[', '{[}')
            str_tag = str_tag.replace(']', '{]}')
            print('{0} & {1} \\\\'.format(str_tag, round(bilu_coefs[bilu_tag], 2)))
    else:
        print('  POS tags: {}'.format(' '.join(sorted_pos)))
        print('  BILU tags: {}'.format(' '.join(sorted_bilu)))
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

    # Run context_logprob with different window sizes and ngram lengths.
    ngrams = [1,2,3,4,5]
    window_sizes = [1,2,4,8,16,32,64,128]
    for target_var in ['surprisal', 'var_steps', 'aoa', 'forgettability', 'var_runs']:
        expected_sign = '+' if target_var == 'surprisal' else '-'
        run_context_regressions(average_df, target_var, annotators[0], ngrams, window_sizes, expected_sign=expected_sign)

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
        directions_of_effect(average_df, target_var, table_format=True)

    # Run regressions and LRTs.
    for target_var in ['surprisal', 'var_steps', 'aoa', 'forgettability', 'var_runs']:
        run_lrts(average_df, target_var)
    return


if __name__ == "__main__":
    main()
