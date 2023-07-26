"""
Along with general curve similarity across runs, we may want to know whether
fine-grained noise is similar across runs. Is there systematicity to the noise?

I.e. do some example pairs consistently have correlated noise, across
pre-training runs?
"""

import sys
sys.path.append('lm-learning-curves')
from utils.annotator import CurveAnnotator

# TODO.
