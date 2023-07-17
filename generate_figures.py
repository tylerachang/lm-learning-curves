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

from utils.annotator import CurveAnnotator

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
              norm=mpl.colors.LogNorm(clip=True, vmin=1, vmax=5000),
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


# Mean correlation across runs for confidence, variability, and AoA.
# Plot correlation for runs 0 and 1.
# Plot performance correlation during pre-training.
# Plot correlation with n-grams during pre-training.
def cross_run_correlations(annotators):
    print('Computing cross-run correlations for confidence, variability, and AoA.')
    # Last 11: last 25% of pre-training.
    # Last 27: last 50% of pre-training.
    last_n = 11

    # Confidence scores.
    scores = []
    for annotator in annotators:
        scores.append(annotator.get_confidence_scores(last_n=last_n))
    correlations = get_correlations(scores)
    print('Confidence cross-run correlation: {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))
    plot_correlation_hist(scores[0], scores[1],
            figname='confidence_crossrun_correlation.pdf',
            xlabel='Run 0 confidence', ylabel='Run 1 confidence')

    # Variability scores.
    scores = []
    for annotator in annotators:
        scores.append(annotator.get_variability_scores(last_n=last_n))
    correlations = get_correlations(scores)
    print('Variability cross-run correlation: {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))
    plot_correlation_hist(scores[0], scores[1],
            figname='variability_crossrun_correlation.pdf',
            xlabel='Run 0 variability', ylabel='Run 1 variability')

    # AoA scores.
    scores = []
    for annotator in annotators:
        scores.append(annotator.get_aoa_values()[:, 0])
    correlations = get_correlations(scores)
    print('AoA cross-run correlation: {0} +/- {1}'.format(
            np.mean(correlations), np.std(correlations)))
    print('Min: {0}, Max: {1}'.format(np.min(correlations), np.max(correlations)))
    # Drop the examples set to min or max step.
    mask = ((scores[0] != np.min(scores[0])) & (scores[0] != np.max(scores[0])) &
            (scores[1] != np.min(scores[1])) & (scores[1] != np.max(scores[1])))
    plot_correlation_hist(scores[0][mask], scores[1][mask],
            figname='aoa_crossrun_correlation.pdf',
            xlabel='Run 0 AoA', ylabel='Run 1 AoA')

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
    fig = plt.figure(figsize=(4, 3))
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
    plt.savefig(os.path.join(FIGURE_DIR, 'surprisal_ngram_correlation.pdf'), bbox_inches='tight')
    # Plot correlation between runs.
    fig = plt.figure(figsize=(4, 3))
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
        ax.text(log10_step, 0.95-ngram_i*0.01, '{}-gram'.format(ngram_i+1),
                horizontalalignment='left', backgroundcolor='white')
    ax.set_ylabel('Cross-run surprisal correlation')
    ax.set_xlabel('Pre-training step (log10)')
    plt.savefig(os.path.join(FIGURE_DIR, 'surprisal_crossrun_correlation.pdf'), bbox_inches='tight')
    return


def dataset_map(annotators):
    print('Plotting dataset map.')
    last_n = 11
    confidence_scores = []
    variability_scores = []
    for annotator in annotators:
        confidence_scores.append(annotator.get_confidence_scores(last_n=last_n))
        variability_scores.append(annotator.get_variability_scores(last_n=last_n))
    confidence_scores = np.concatenate(confidence_scores, axis=0)
    variability_scores = np.concatenate(variability_scores, axis=0)
    # Convert from suprisal to raw probability.
    confidence_scores = np.power(2, -1.0*confidence_scores)
    plot_correlation_hist(confidence_scores, variability_scores, figname='dataset_map.pdf',
                  xlabel='Confidence (P)', ylabel='Variability (stdev(surprisal))')
    return


def main():
    annotators_dir = 'annotators'
    annotators = []
    for run_i in range(5):
        annotator = CurveAnnotator(os.path.join(annotators_dir, 'gpt2_{}'.format(run_i)))
        annotators.append(annotator)
    # Run analyses.
    dataset_map(annotators)
    cross_run_correlations(annotators)


if __name__ == "__main__":
    main()
