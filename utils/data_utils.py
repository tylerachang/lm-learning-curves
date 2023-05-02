"""
Data utilities for the monolingual pre-training analyses.
"""

import os
import math
import codecs
import numpy as np

# Returns a list of integers (checkpoint steps).
def get_checkpoints(model_dir):
    checkpoints = []
    for dir_name in os.listdir(model_dir):
        if not os.path.isdir(os.path.join(model_dir, dir_name)):
            continue
        if dir_name[:10] != "checkpoint":
            continue
        try:
            checkpoint = int(dir_name.split("-")[-1])
            checkpoints.append(checkpoint)
        except:
            print("Not a checkpoint, skipping: {}".format(os.path.join(model_dir, dir_name)))
    checkpoints.sort()
    return checkpoints

# Returns a list of lists of integers (token ids), given an input text file
# of tokenized examples (space-separated token ids).
def get_examples(filepath, max_examples=-1):
    if max_examples == -1:
        max_examples = math.inf
    # Load examples.
    total_tokens = 0
    examples = []
    examples_file = codecs.open(filepath, 'rb', encoding='utf-8')
    for line in examples_file:
        if len(examples) >= max_examples:
            break
        stripped_line = line.strip()
        if stripped_line == "":
            continue
        example = [int(token_id) for token_id in stripped_line.split()]
        total_tokens += len(example)
        examples.append(example)
    examples_file.close()
    return examples
