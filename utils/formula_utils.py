"""
Formulas for the monolingual pre-training analyses.
"""

import numpy as np
from scipy.optimize import curve_fit

# Convert a checkpoint number to time step.
def exponential_checkpoint_to_step(checkpoint_n, s0, s1, t1):
    term1 = s0 * t1 / (s1 - s0)
    exponent = checkpoint_n * (s1 - s0) / t1
    term2 = np.exp(exponent) - 1.0
    return term1 * term2

# Convert a time step to checkpoint number.
def exponential_step_to_checkpoint(step_n, s0, s1, t1):
    term1 = t1 / (s1 - s0)
    term2 = 1.0 + (s1 - s0) * step_n / (s0 * t1)
    term2 = np.log(term2)
    return term1 * term2

# Get the checkpoint steps given a number of max steps.
def get_checkpoint_steps(s0, s1, t1, max_steps):
  checkpoint_steps = []
  checkpoint_step = 0
  n = 0
  while checkpoint_step <= max_steps:
    checkpoint_steps.append(checkpoint_step)
    # Increment.
    n += 1
    checkpoint_step = exponential_checkpoint_to_step(n)
    checkpoint_step = int(np.round(checkpoint_step, 0))
  return checkpoint_steps

# Sigmoid regression and AoA utilities.
# Note: start and end are the two asymptotes.
def sigmoid(x, start, end, xmid, scale):
    asymptote = end - start
    exponent = (xmid - x) / scale
    return start + asymptote / (1.0 + np.exp(exponent))
# Ideally, an increasing sigmoid.
def run_sigmoid_regression(xdata, ydata, max_iters=1000000):
    # Relative error tolerance.
    ftol = 1e-08
    succeeded = False
    # Try with increasing tolerance. Print warning if not fitted on first attempt.
    while not succeeded:
        try:
            popt, pcov = curve_fit(sigmoid, xdata, ydata,
                    p0=[np.min(ydata), np.max(ydata), np.mean(xdata), 1.0],
                    maxfev=max_iters, ftol=ftol)
            succeeded = True
        except RuntimeError:
            print('WARNING: sigmoid curve_fit failed for ftol={}'.format(ftol))
        ftol = ftol * 10.0
    return popt
# Sigmoid params: (start, end, xmid, scale).
# Finds where the fitted sigmoid passes through proportion between ymin and ymax.
# Returns the x and y value.
def get_aoa(ymin, ymax, proportion, xmin, xmax, sigmoid_params):
    start, end, xmid, scale = tuple(sigmoid_params)
    ythreshold = ymin*(1-proportion) + ymax*proportion
    # Set to maximum x value if:
    # (1) ythreshold is higher than the entire curve,
    # (2) ymin is greater than ymax,
    # (3) the sigmoid is decreasing instead of increasing.
    if ((ythreshold >= start and ythreshold >= end) or
        (ymin >= ymax) or
        (end < start and scale > 0) or (end > start and scale < 0)):
        return xmax, None
    # Set to minimum x value if:
    # (1) ythreshold is lower than the entire curve.
    if (ythreshold <= start and ythreshold <= end):
        return xmin, None
    # Compute the intersection between the fitted sigmoid and ythreshold.
    intersectionx = xmid - scale * np.log((end-start)/(ythreshold-start) - 1.0)
    return intersectionx, ythreshold
# Returns: (aoa, surprisal_threshold, start, end, xmid, scale).
# surprisal_threshold may be -1.0 if the AoA was set to the minimum or maximum step.
# chance_surprisal is generally -1.0*np.log2(1/vocab_size).
# start, end, xmid, and scale are the fitted sigmoid parameters.
# Note: the first checkpoint (step 0) should not be included, because the log step
# is negative infinity.
def compute_aoa_vals(log_steps, surprisals, chance_surprisal, proportion=0.50):
    # Sigmoid curves should generally be increasing.
    negative_surprisals = -1.0 * surprisals
    negative_chance = -1.0 * chance_surprisal
    sigmoid_params = run_sigmoid_regression(log_steps, negative_surprisals)
    ymin = negative_chance
    ymax = np.max(negative_surprisals)
    xmin = np.min(log_steps)
    xmax = np.max(log_steps)
    aoa, surprisal_threshold = get_aoa(ymin, ymax, proportion, xmin, xmax, sigmoid_params)
    # Convert back to positive surprisal values.
    surprisal_threshold = -1.0 if surprisal_threshold is None else -1.0 * surprisal_threshold
    sigmoid_params[0] = -1.0 * sigmoid_params[0]
    sigmoid_params[1] = -1.0 * sigmoid_params[1]
    return (aoa, surprisal_threshold, *sigmoid_params)

# Fine-grained trajectory utilities.
# Returns the slope for each window, with some stride.
# Output shapes: (n_windows). With window_size=1, we have n_windows = n_surprisals-window_size+1.
# Note: the first checkpoint (step 0) should not be included, because the log step
# is negative infinity.
def get_curve_slopes(log_steps, surprisals, window_size=10, stride=1):
    n_vals = log_steps.shape[0]
    slopes = []
    for start_i in range(0, n_vals-window_size+1, stride):
        # Compute slope.
        reg = LinearRegression(fit_intercept=True).fit(log_steps[start_i:start_i+window_size].reshape(-1, 1),
                                                       surprisals[start_i:start_i+window_size])
        slope = reg.coef_
        slopes.append(slope)
    return np.array(slopes)
