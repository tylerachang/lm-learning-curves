"""
Plot figures and get correlations.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import itertools
import scipy

import sys
sys.path.append('lm-learning-curves')
from utils.annotator import CurveAnnotator, get_features_dataframe, get_average_features_dataframe, get_crossrun_variability

FIGURE_DIR = 'figures'


# Plot a correlation 2D histogram.
def plot_correlation_hist(x, y, figname='figure.pdf',
              xlabel='', ylabel='', clip_std=5.0,
              figsize=(3,3), dpi=300):
    # Clipping.
    xmean = np.mean(x)
    xstd = np.std(x)
    xmin = xmean - clip_std * xstd
    xmax = xmean + clip_std * xstd
    ymean = np.mean(y)
    ystd = np.std(y)
    ymin = ymean - clip_std * ystd
    ymax = ymean + clip_std * ystd
    mask = (xmin <= x) & (x <= xmax) & (ymin <= y) & (y <= ymax)
    # Plotting.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.hist2d(x[mask], y[mask], bins=[100, 100],
              norm=mpl.colors.LogNorm(clip=True, vmin=1, vmax=10000),
              cmap='Purples')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(os.path.join(FIGURE_DIR, figname), dpi=dpi, bbox_inches='tight')
    return


# Input: list of score vectors.
# Output: list of pairwise correlations.
def get_correlations(scores):
    pairs = itertools.combinations(range(len(scores)), 2)
    correlations = []
    for i, j in pairs:
        r, p = scipy.stats.pearsonr(scores[i], scores[j])
        correlations.append(r)
    return correlations


# Mean correlation across runs for surprisal, variability (steps), AoA, and forgettability.
# Plot correlation for runs 0 and 1.
def crossrun_correlations(annotators):
    print('Computing cross-run correlations for surprisal, variability (steps), AoA, and forgettability.')
    # Last 11: last 25% of pre-training.
    # Last 27: last 50% of pre-training.
    last_n = 11

    # Surprisal (related to inverse confidence) scores.
    scores = []
    for annotator in annotators:
        scores.append(annotator.get_confidence_scores(last_n=last_n))
    correlations = get_correlations(scores)
    print('Surprisal cross-run correlation: {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))
    plot_correlation_hist(scores[0], scores[1],
            figname='final_surprisal_crossrun_correlation.pdf',
            xlabel='Run 0 surprisal', ylabel='Run 1 surprisal')

    # Variability (across steps) scores.
    scores = []
    for annotator in annotators:
        scores.append(annotator.get_variability_scores(last_n=last_n))
    correlations = get_correlations(scores)
    print('Variability (steps) cross-run correlation: {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))
    plot_correlation_hist(scores[0], scores[1],
            figname='var_steps_crossrun_correlation.pdf',
            xlabel='Run 0 var (steps)', ylabel='Run 1 var (steps)')

    # GAM AoA scores.
    scores = []
    for annotator in annotators:
        scores.append(annotator.get_gam_aoas()[:, 0])
    correlations = get_correlations(scores)
    print('GAM AoA cross-run correlation: {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))
    # Drop the examples set to min or max step.
    mask = ((scores[0] != np.min(scores[0])) & (scores[0] != np.max(scores[0])) &
            (scores[1] != np.min(scores[1])) & (scores[1] != np.max(scores[1])))
    plot_correlation_hist(scores[0][mask], scores[1][mask],
            figname='aoa_crossrun_correlation.pdf',
            xlabel='Run 0 AoA', ylabel='Run 1 AoA')

    # Forgettability scores.
    scores = []
    for annotator in annotators:
        scores.append(annotator.get_forgettability_scores())
    correlations = get_correlations(scores)
    print('Forgettability cross-run correlation: {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))
    # Drop the examples set to min or max step.
    plot_correlation_hist(scores[0][mask], scores[1][mask],
            figname='forgettability_crossrun_correlation.pdf',
            xlabel='Run 0 forgettability', ylabel='Run 1 forgettability')

    # Variability (across runs).
    crossrun_dists = get_crossrun_variability(annotators, use_gams=True)
    # Compute correlation when using different subsets of the pre-training runs.
    # Consider all possible ways to assign three runs to one group, and two
    # runs to the other. Compute cross-run variability for each group, and get
    # correlation.
    correlations = []
    size3_subsets = list(itertools.combinations(range(len(annotators)), 3))
    orig_pairs = list(itertools.combinations(range(len(annotators)), 2))
    for size3_subset in size3_subsets:
        # Pairwise distances within the subset.
        dist_indices = [pair_i for pair_i, pair in enumerate(orig_pairs) if (pair[0] in size3_subset) and (pair[1] in size3_subset)]
        var1 = np.mean(crossrun_dists[:, dist_indices], axis=-1)
        # Pairwise distances outside the subset.
        dist_indices = [pair_i for pair_i, pair in enumerate(orig_pairs) if (pair[0] not in size3_subset) and (pair[1] not in size3_subset)]
        var2 = np.mean(crossrun_dists[:, dist_indices], axis=-1)
        r, p = scipy.stats.pearsonr(var1, var2)
        correlations.append(r)
    print('Variability (runs) cross-run-subset correlation (disjoint): {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))
    # Instead, consider all 4-run subsets.
    # Pairs of subsets, each subset defined by the run it drops.
    subset_pairs = itertools.combinations(range(len(annotators)), 2)
    correlations = []
    orig_pairs = list(itertools.combinations(range(len(annotators)), 2))
    for drop_i, drop_j in subset_pairs:
        # Pairwise distances within the first subset.
        dist_indices = [pair_i for pair_i, pair in enumerate(orig_pairs) if (pair[0] != drop_i) and (pair[1] != drop_i)]
        var1 = np.mean(crossrun_dists[:, dist_indices], axis=-1)
        # Pairwise distances within the second subset.
        dist_indices = [pair_i for pair_i, pair in enumerate(orig_pairs) if (pair[0] != drop_j) and (pair[1] != drop_j)]
        var2 = np.mean(crossrun_dists[:, dist_indices], axis=-1)
        r, p = scipy.stats.pearsonr(var1, var2)
        correlations.append(r)
    print('Variability (runs) cross-run-subset correlation (4-run subsets): {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))

    # Correlation between GAM distance and raw surprisal curve distance.
    gams_dists = get_crossrun_variability(annotators, use_gams=True).flatten()
    raw_dists = get_crossrun_variability(annotators, use_gams=False).flatten()
    r, p = scipy.stats.pearsonr(gams_dists, raw_dists)
    print('Correlation between GAMS cross-run distance and raw cross-run distance: r={}'.format(r))
    # Correlation between GAM-based cross-run variability and raw cross-run variability.
    gams_variability = np.mean(get_crossrun_variability(annotators, use_gams=True), axis=-1)
    raw_variability = np.mean(get_crossrun_variability(annotators, use_gams=False), axis=-1)
    r, p = scipy.stats.pearsonr(gams_variability, raw_variability)
    print('Correlation between GAMS cross-run var and raw cross-run var: r={}'.format(r))
    return


# Plot performance correlation during pre-training.
# Plot correlation with n-grams during pre-training.
def crossrun_surprisal_correlations(annotators):
    # Surprisals during pre-training.
    print('Computing cross-run surprisal correlations during pre-training.')
    log10_steps = annotators[0].get_log10_steps()
    surprisal_curves = []
    for annotator in annotators:
        surprisal_curves.append(annotator.get_surprisal_curves())
    # Mean and stdev correlation between runs.
    means = np.zeros(len(log10_steps))
    stdevs = np.zeros(len(log10_steps))
    for checkpoint_i in tqdm(range(len(log10_steps))):
        correlations = get_correlations([curves[:, checkpoint_i] for curves in surprisal_curves])
        means[checkpoint_i] = np.mean(correlations)
        stdevs[checkpoint_i] = np.std(correlations)
        if stdevs[checkpoint_i] > 0.001:
            print('STDEV > 0.001: step {}'.format(np.power(10, log10_steps[checkpoint_i])))
    # Correlation with n-gram surprisal for 1 <= n <= 5.
    ngram_corr_means = np.zeros((5, len(log10_steps)))
    ngram_corr_stdevs = np.zeros((5, len(log10_steps)))
    for ngram_i in range(5):
        print('Computing correlation with {}-gram surprisals.'.format(ngram_i+1))
        ngram_surprisals = annotators[0].get_target_ngram_surprisals('full_train', ngram_i+1)
        for checkpoint_i in tqdm(range(len(log10_steps))):
            corrs = []
            for run_i in range(5):
                r, p = scipy.stats.pearsonr(surprisal_curves[run_i][:, checkpoint_i], ngram_surprisals)
                corrs.append(r)
            ngram_corr_means[ngram_i, checkpoint_i] = np.mean(corrs)
            ngram_corr_stdevs[ngram_i, checkpoint_i] = np.std(corrs)
    # Plot correlation with n-grams.
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot()
    for ngram_i in range(5):
        ax.plot(log10_steps, ngram_corr_means[ngram_i], label='{}-gram'.format(ngram_i+1))
        ax.fill_between(log10_steps,
                        ngram_corr_means[ngram_i]-5*ngram_corr_stdevs[ngram_i],
                        ngram_corr_means[ngram_i]+5*ngram_corr_stdevs[ngram_i],
                        alpha=0.25, linewidth=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_ylabel('Correlation with n-gram surprisal')
    ax.set_xlabel('Pre-training step (log10)')
    ax.set_xlim(2.0, 6.0)
    plt.savefig(os.path.join(FIGURE_DIR, 'surprisal_ngram_correlation.pdf'), bbox_inches='tight')
    # Plot correlation between runs.
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot()
    # Outlier correlation:
    # Step 451441, run 4 has correlation ~0.962 with other runs instead of 0.973.
    # Run 4 mean surprisal: 5.228591488130975
    # Other runs: 5.187595936132901, 5.189338698873389, 5.189314655430209, 5.190382975574222.
    ax.plot(log10_steps, means, color='blueviolet')
    ax.fill_between(log10_steps, means-5*stdevs, means+5*stdevs, alpha=0.25, color='blueviolet', linewidth=0)
    # Vertical lines for maximal similarity to n-gram.
    for ngram_i in range(5):
        log10_step = log10_steps[np.argmax(ngram_corr_means[ngram_i])]
        ax.axvline(x=log10_step, linestyle='dashed', color='mediumblue', alpha=0.60)
        t = ax.text(log10_step, 0.95-ngram_i*0.01, '{}-gram'.format(ngram_i+1),
                horizontalalignment='left')
        t.set_bbox(dict(facecolor='white', alpha=0.50, edgecolor='white'))
    ax.set_ylabel('Cross-run surprisal correlation')
    ax.set_xlabel('Pre-training step (log10)')
    ax.set_xlim(2.0, 6.0)
    plt.savefig(os.path.join(FIGURE_DIR, 'surprisal_crossrun_correlation.pdf'), bbox_inches='tight')
    return


def dataset_map(annotators):
    print('Plotting dataset map.')
    last_n = 11
    confidence_scores = []
    variability_scores = []
    for annotator in annotators:
        prob_curves = annotator.get_surprisal_curves()
        # Convert from suprisal to raw probability.
        prob_curves = np.power(2, -1.0*prob_curves)
        variability_scores.append(np.std(prob_curves[:, -last_n:], axis=-1))
        confidence_scores.append(np.mean(prob_curves[:, -last_n:], axis=-1))
    confidence_scores = np.concatenate(confidence_scores, axis=0)
    variability_scores = np.concatenate(variability_scores, axis=0)
    plot_correlation_hist(variability_scores, confidence_scores, figname='dataset_map.pdf',
                  xlabel='Variability (stdev(P))', ylabel='Confidence (mean(P))',
                  figsize=(4,3))
    return


def cross_metric_correlations(annotators):
    # All within-run metrics, concatenating scores for all runs.
    dataframe = get_features_dataframe(annotators)
    print('Concatenating scores for all runs (except var_runs):')
    print(dataframe[['surprisal', 'var_steps', 'aoa', 'forgettability', 'var_runs']].corr())
    del dataframe
    # All metrics, averaging across runs.
    dataframe = get_average_features_dataframe(annotators)
    print('Averaging scores for all runs:')
    print(dataframe[['surprisal', 'var_steps', 'aoa', 'forgettability', 'var_runs']].corr())
    del dataframe
    return


def plot_contextual_diversity(annotator):
    # Plot raw contextual diversity vs. unigram frequency.
    return


def main():
    annotators_dir = 'annotators'
    annotators = []
    for run_i in range(5):
        annotator = CurveAnnotator(os.path.join(annotators_dir, 'gpt2_{}'.format(run_i)))
        annotators.append(annotator)
    # Plot contextual diversity adjustment.
    plot_contextual_diversity(annotators[0])
    # Correlations between the four metrics.
    cross_metric_correlations(annotators)
    # Plot dataset map (confidence vs. variability).
    dataset_map(annotators)
    # Get correlations for each metric across runs.
    crossrun_correlations(annotators)
    # Get surprisal correlations across runs, over pre-training steps.
    crossrun_surprisal_correlations(annotators)


if __name__ == "__main__":
    main()
